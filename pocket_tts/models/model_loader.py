"""Factory and checkpoint loading for the Pocket TTS model."""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import safetensors
import safetensors.torch
import torch

from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.models.mimi import build_mimi
from pocket_tts.modules.stateful_module import StatefulModule
from pocket_tts.quantization import RECOMMENDED_CONFIG, apply_dynamic_int8
from pocket_tts.utils.config import Config
from pocket_tts.utils.utils import download_if_necessary, size_of_dict
from pocket_tts.utils.weights_loading import (
    get_flow_lm_state_dict,
    get_mimi_state_dict,
    get_training_checkpoint_state_dicts,
)

if TYPE_CHECKING:
    from pocket_tts.models.tts_model import TTSModel

logger = logging.getLogger(__name__)


def quantize_model(tts_model: "TTSModel") -> None:
    """Apply dynamic INT8 quantization to the flow LM attention and FFN layers."""
    apply_dynamic_int8(tts_model.flow_lm, RECOMMENDED_CONFIG)


def stamp_state_names(tts_model: "TTSModel") -> None:
    """StatefulModules find their slice of the state dict by absolute name."""
    top_modules: tuple[torch.nn.Module, ...] = (tts_model.flow_lm, tts_model.mimi)
    for top_module in top_modules:
        for module_name, module in top_module.named_modules():
            if isinstance(module, StatefulModule):
                module._module_absolute_name = module_name


def build_flow_lm(config: Config) -> FlowLMModel:
    """Build the FlowLM from a config; weights are loaded on top of the fresh one."""
    return FlowLMModel.from_pydantic_config(
        config.flow_lm,
        latent_dim=config.mimi.quantizer.dimension,
        insert_bos_before_voice=config.flow_lm.insert_bos_before_voice,
    )


def new_tts_model(
    config: Config,
    temp: float,
    sampler_decode_steps: int,
    noise_clamp: float | None,
    eos_threshold: float,
    origin: Path | None,
) -> "TTSModel":
    """Build a fresh, untrained TTSModel from a config, for weights or checkpoints to load into."""
    from pocket_tts.models.tts_model import TTSModel

    flow_lm = build_flow_lm(config)
    return TTSModel(
        flow_lm,
        temp,
        sampler_decode_steps,
        noise_clamp,
        eos_threshold,
        config,
        origin=origin,
        pad_with_spaces_for_short_inputs=config.pad_with_spaces_for_short_inputs,
        model_recommended_frames_after_eos=config.model_recommended_frames_after_eos,
        remove_semicolons=config.remove_semicolons,
        append_terminal_punctuation=config.append_terminal_punctuation,
        capitalize_first_letter=config.capitalize_first_letter,
        replace_characters=config.replace_characters,
    )


def load_training_checkpoint(tts_model: "TTSModel", path: str | Path, use_ema: bool = True) -> None:
    """Load weights from a training checkpoint (.pt) into a config-built model."""
    tts_model.has_custom_weights = True
    flow_lm_state, mimi_state = get_training_checkpoint_state_dicts(Path(path), use_ema)
    config = tts_model.config
    tts_model.flow_lm.speaker_proj_weight = torch.nn.Parameter(
        torch.zeros(
            (
                config.flow_lm.transformer.d_model,
                config.mimi.inner_dim or config.mimi.seanet.dimension,
            ),
            dtype=torch.float32,
        )
    )
    tts_model.flow_lm.load_state_dict(flow_lm_state, strict=True)
    tts_model.mimi = build_mimi(config.mimi).to(device="cpu")
    # Training freezes Mimi and does not checkpoint it, so its weights come
    # from the config -- either its own safetensors or the packaged bundle.
    if mimi_state:
        tts_model.mimi.load_state_dict(mimi_state, strict=False)
    elif config.mimi.weights_path is not None:
        tts_model.mimi.load_state_dict(
            get_mimi_state_dict(download_if_necessary(config.mimi.weights_path)), strict=True
        )
    else:
        if config.weights_path is None:
            raise ValueError("no Mimi weights: set mimi.weights_path or weights_path")
        try:
            bundle = download_if_necessary(config.weights_path)
        except Exception:
            if config.weights_path_without_voice_cloning is None:
                raise
            tts_model.has_voice_cloning = False
            bundle = download_if_necessary(config.weights_path_without_voice_cloning)
        bundled = safetensors.torch.load_file(bundle)
        tts_model.mimi.load_state_dict(
            {k.removeprefix("mimi."): v for k, v in bundled.items() if k.startswith("mimi.")},
            strict=True,
        )
    tts_model.mimi.eval()
    stamp_state_names(tts_model)


def load_flow_lm_weights(tts_model: "TTSModel", config: Config) -> None:
    """Load the FlowLM weights (and its voice-cloning projection) from the config."""
    tts_model.flow_lm.speaker_proj_weight = torch.nn.Parameter(
        torch.zeros(
            (
                config.flow_lm.transformer.d_model,
                config.mimi.inner_dim or config.mimi.seanet.dimension,
            ),
            dtype=torch.float32,
        )
    )
    if config.flow_lm.weights_path is None:
        return
    if config.mimi.weights_path is None:
        raise ValueError("If you specify flow_lm.weights_path you should specify mimi.weights_path")
    logger.info(f"Loading FlowLM weights from {config.flow_lm.weights_path}")
    state_dict_flowlm = get_flow_lm_state_dict(download_if_necessary(config.flow_lm.weights_path))
    tts_model.flow_lm.load_state_dict(state_dict_flowlm, strict=True)


def load_mimi_weights(tts_model: "TTSModel", config: Config) -> None:
    """Load the Mimi weights from the config."""
    tts_model.mimi = build_mimi(config.mimi).to(device="cpu")
    if config.mimi.weights_path is None:
        return
    if config.flow_lm.weights_path is None and config.weights_path is None:
        raise ValueError(
            "If you specify mimi.weights_path you should specify flow_lm.weights_path "
            "or weights_path"
        )
    logger.info(f"Loading Mimi weights from {config.mimi.weights_path}")
    mimi_state = get_mimi_state_dict(download_if_necessary(config.mimi.weights_path))
    tts_model.mimi.load_state_dict(mimi_state, strict=True)
    tts_model.mimi.eval()


def load_tts_weights(tts_model: "TTSModel", config: Config) -> None:
    """Load the full-model weights; a config Mimi weights file replaces the bundle's Mimi."""
    if config.weights_path is not None:
        logger.info(f"Loading TTSModel weights from {config.weights_path}")
        try:
            weights_file = download_if_necessary(config.weights_path)
        except Exception:
            if config.weights_path_without_voice_cloning is None:
                raise
            tts_model.has_voice_cloning = False
            weights_file = download_if_necessary(config.weights_path_without_voice_cloning)

        state_dict = safetensors.torch.load_file(weights_file)
        if config.mimi.weights_path is not None:
            # mimi.weights_path, loaded above, replaces the Mimi weights of the bundle.
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("mimi.")}
            state_dict.update({f"mimi.{k}": v for k, v in tts_model.mimi.state_dict().items()})
        tts_model.load_state_dict(state_dict, strict=True)

    if config.flow_lm.weights_path is None and config.weights_path is None:
        logger.warning("No weights_path specified for FlowLM or TTSModel, model is uninitialized!")
    size_in_mb = size_of_dict(tts_model.state_dict()) // 1e6
    if os.environ.get("POCKET_TTS_SAVE_WEIGHTS", "0") == "1":
        save_path = "./model.safetensors"
        safetensors.torch.save_file(tts_model.state_dict(), save_path)
        logger.info(f"Saved TTSModel weights to {save_path}")
    logging.info(f"TTS Model loaded successfully. Its size is {size_in_mb} MB")


def _from_pydantic_config(
    config: Config,
    temp: float,
    sampler_decode_steps: int,
    noise_clamp: float | None,
    eos_threshold: float,
    origin: Path | None,
) -> "TTSModel":
    """Build a fresh, untrained TTSModel from a config, for weights or checkpoints to load into."""
    return new_tts_model(config, temp, sampler_decode_steps, noise_clamp, eos_threshold, origin)


def _from_pydantic_config_with_weights(
    config: Config,
    temp: float,
    sampler_decode_steps: int,
    noise_clamp: float | None,
    eos_threshold: float,
    origin: Path | None = None,
) -> "TTSModel":
    """Build a config-based TTSModel and load its weights (from a checkpoint or the config)."""
    tts_model = _from_pydantic_config(
        config, temp, sampler_decode_steps, noise_clamp, eos_threshold, origin=origin
    )
    tts_model.flow_lm.speaker_proj_weight = torch.nn.Parameter(
        torch.zeros(
            (
                config.flow_lm.transformer.d_model,
                config.mimi.inner_dim or config.mimi.seanet.dimension,
            ),
            dtype=torch.float32,
        )
    )
    if config.flow_lm.weights_path is not None:
        if config.mimi.weights_path is None:
            raise ValueError(
                "If you specify flow_lm.weights_path you should specify mimi.weights_path"
            )
        logger.info(f"Loading FlowLM weights from {config.flow_lm.weights_path}")
        state_dict_flowlm = get_flow_lm_state_dict(
            download_if_necessary(config.flow_lm.weights_path)
        )
        tts_model.flow_lm.load_state_dict(state_dict_flowlm, strict=True)

    tts_model.mimi = build_mimi(config.mimi).to(device="cpu")

    # Load mimi weights from the config safetensors file with complete mapping for strict loading

    if config.mimi.weights_path is not None:
        if config.flow_lm.weights_path is None and config.weights_path is None:
            raise ValueError(
                "If you specify mimi.weights_path you should specify flow_lm.weights_path "
                "or weights_path"
            )
        logger.info(f"Loading Mimi weights from {config.mimi.weights_path}")
        mimi_state = get_mimi_state_dict(download_if_necessary(config.mimi.weights_path))
        tts_model.mimi.load_state_dict(mimi_state, strict=True)

    tts_model.mimi.eval()

    if config.weights_path is not None:
        logger.info(f"Loading TTSModel weights from {config.weights_path}")
        try:
            weights_file = download_if_necessary(config.weights_path)
        except Exception:
            if config.weights_path_without_voice_cloning is None:
                raise
            tts_model.has_voice_cloning = False
            weights_file = download_if_necessary(config.weights_path_without_voice_cloning)

        state_dict = safetensors.torch.load_file(weights_file)
        if config.mimi.weights_path is not None:
            # mimi.weights_path, loaded above, replaces the Mimi weights of the bundle.
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("mimi.")}
            state_dict.update({f"mimi.{k}": v for k, v in tts_model.mimi.state_dict().items()})
        tts_model.load_state_dict(state_dict, strict=True)

    if config.flow_lm.weights_path is None and config.weights_path is None:
        logger.warning("No weights_path specified for FlowLM or TTSModel, model is uninitialized!")
    size_in_mb = size_of_dict(tts_model.state_dict()) // 1e6
    if os.environ.get("POCKET_TTS_SAVE_WEIGHTS", "0") == "1":
        save_path = "./model.safetensors"
        safetensors.torch.save_file(tts_model.state_dict(), save_path)
        logger.info(f"Saved TTSModel weights to {save_path}")
    logging.info(f"TTS Model loaded successfully. Its size is {size_in_mb} MB")

    stamp_state_names(tts_model)
    return tts_model


def load_config_built_model(
    config: Config,
    temp: float,
    sampler_decode_steps: int,
    noise_clamp: float | None,
    eos_threshold: float,
    origin: Path | None,
    checkpoint: str | Path | None = None,
) -> "TTSModel":
    """Build a config-based TTSModel and load its weights (from a checkpoint or the config)."""
    tts_model = new_tts_model(
        config, temp, sampler_decode_steps, noise_clamp, eos_threshold, origin
    )
    if checkpoint is not None:
        load_training_checkpoint(tts_model, checkpoint)
    else:
        load_flow_lm_weights(tts_model, config)
        load_mimi_weights(tts_model, config)
        load_tts_weights(tts_model, config)
    stamp_state_names(tts_model)
    return tts_model
