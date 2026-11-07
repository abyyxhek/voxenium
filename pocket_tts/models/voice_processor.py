"""Voice conditioning: encode audio prompts into flow-model latents and build voice states."""

import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import scipy.io.wavfile
import torch
from torch.nn import functional as F

from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio, end_on_pause
from pocket_tts.models.flow_engine import run_flow_lm_and_increment_step
from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.models.mimi import MimiModel
from pocket_tts.models.model_state import _import_model_state, _is_safetensors_source
from pocket_tts.modules.stateful_module import ModelState, init_states
from pocket_tts.utils.config import CONFIGS_DIR, Config
from pocket_tts.utils.utils import (
    _ORIGINS_OF_PREDEFINED_VOICES,
    DEBUG_MIMI,
    display_execution_time,
    download_if_necessary,
    get_predefined_voice,
    size_of_dict,
)


if TYPE_CHECKING:
    from pocket_tts.models.tts_model import TTSModel

logger = logging.getLogger(__name__)

VOICE_CLONING_UNSUPPORTED = (
    f"We could not download the weights for the model with voice cloning, "
    f"but you're trying to use voice cloning. "
    f"Without voice cloning, you can use our catalog of voices {list(_ORIGINS_OF_PREDEFINED_VOICES.keys())}. "
    f"If you want access to the model with voice cloning, go to "
    f"https://huggingface.co/kyutai/pocket-tts and accept the terms, "
    f"then make sure you're logged in locally with `uvx hf auth login`."
)


def decode_and_dump(mimi: MimiModel, encoded: torch.Tensor, filename: str) -> None:
    mimi_state = init_states(mimi, batch_size=1, sequence_length=10000)
    resored_audio = mimi.decode_from_latent(encoded, mimi_state)
    scipy.io.wavfile.write(filename, mimi.sample_rate, resored_audio.numpy())
    logger.info("Saved restored audio from Mimi encoding to %s for debugging", filename)


def encode_prompt(
    flow_lm: FlowLMModel,
    mimi: MimiModel,
    audio_conditioning: torch.Tensor,
    debug_mimi: bool = False,
) -> torch.Tensor:
    """Encode an audio prompt into the flow model's conditioning latents."""
    encoded = mimi.encode_to_latent(audio_conditioning)
    if debug_mimi:
        # sanity check
        decode_and_dump(mimi, encoded, "debug_encoded_latent_decoded.wav")

    latents = encoded.to(torch.float32)
    conditioning = F.linear(latents, flow_lm.speaker_proj_weight)
    return conditioning


def get_state_for_audio_prompt(
    flow_lm: FlowLMModel,
    mimi: MimiModel,
    audio_conditioning: Path | str | torch.Tensor,
    *,
    device: torch.device,
    has_voice_cloning: bool,
    has_custom_weights: bool,
    origin: Path | None,
    config: Config,
    truncate: bool = False,
    sampler_decode_steps: int,
    temp: float,
    noise_clamp: float | None,
    eos_threshold: float,
) -> ModelState:
    """Create model state conditioned on audio prompt for continuation."""
    if isinstance(audio_conditioning, (str, Path)) and _is_safetensors_source(audio_conditioning):
        if isinstance(audio_conditioning, str):
            audio_conditioning = download_if_necessary(audio_conditioning)

        return _import_model_state(audio_conditioning, device)

    if isinstance(audio_conditioning, str) and audio_conditioning in _ORIGINS_OF_PREDEFINED_VOICES:
        # We get the audio conditioning directly from the safetensors file.
        if has_custom_weights:
            raise ValueError(
                f"Predefined voice {audio_conditioning!r} is a state precomputed with the "
                "released weights; feeding it to other weights leaves the model out of "
                "distribution (it typically never emits EOS). Pass the voice as an audio "
                "file instead."
            )
        if origin is None or not origin.is_relative_to(CONFIGS_DIR):
            raise ValueError(
                f"Cannot use predefined voices when the model "
                f"is not loaded from a config associated with a language."
                f"Here the origin is {origin}"
            )
        return _import_model_state(
            download_if_necessary(
                get_predefined_voice(language=origin.stem, name=audio_conditioning)
            ),
            device,
        )

    if not has_voice_cloning and isinstance(audio_conditioning, (str, Path)):
        raise ValueError(VOICE_CLONING_UNSUPPORTED)

    if isinstance(audio_conditioning, str):
        audio_conditioning = download_if_necessary(audio_conditioning)

    if isinstance(audio_conditioning, Path):
        audio, conditioning_sample_rate = audio_read(audio_conditioning)

        if truncate:
            max_samples = int(30 * conditioning_sample_rate)  # 30 seconds of audio
            if audio.shape[-1] > max_samples:
                audio = audio[..., :max_samples]
                logger.info(f"Audio truncated to first 30 seconds ({max_samples} samples)")

        audio_conditioning = convert_audio(
            audio, conditioning_sample_rate, config.mimi.sample_rate, 1
        )

    audio_conditioning = end_on_pause(audio_conditioning, config.mimi.sample_rate)
    with display_execution_time("Encoding audio prompt"):
        prompt = encode_prompt(
            flow_lm, mimi, audio_conditioning.unsqueeze(0).to(device), debug_mimi=DEBUG_MIMI
        )

    if flow_lm.insert_bos_before_voice:
        prompt = torch.cat([flow_lm.bos_before_voice, prompt], dim=1)

    model_state = init_states(flow_lm, batch_size=1, sequence_length=prompt.shape[1])

    with display_execution_time("Prompting audio"):
        run_flow_lm_and_increment_step(
            flow_lm,
            model_state,
            audio_conditioning=prompt,
            sampler_decode_steps=sampler_decode_steps,
            temp=temp,
            noise_clamp=noise_clamp,
            eos_threshold=eos_threshold,
        )

    logger.info("Size of the model state for audio prompt: %d MB", size_of_dict(model_state) // 1e6)

    return model_state


@lru_cache(maxsize=2)
def _cached_get_state_for_audio_prompt(
    tts_model: "TTSModel", audio_conditioning: Path | str | torch.Tensor, truncate: bool = False
) -> ModelState:
    """Memoize the voice states: encoding an audio prompt is the expensive part."""
    return get_state_for_audio_prompt(
        tts_model.flow_lm,
        tts_model.mimi,
        audio_conditioning,
        device=tts_model.device,
        has_voice_cloning=tts_model.has_voice_cloning,
        has_custom_weights=tts_model.has_custom_weights,
        origin=tts_model.origin,
        config=tts_model.config,
        truncate=truncate,
        sampler_decode_steps=tts_model.sampler_decode_steps,
        temp=tts_model.temp,
        noise_clamp=tts_model.noise_clamp,
        eos_threshold=tts_model.eos_threshold,
    )


def load_voice(
    tts_model: "TTSModel", audio_conditioning: Path | str | torch.Tensor, truncate: bool = False
) -> ModelState:
    """Load the voice state of an audio prompt, memoizing the computed speaker latent states."""
    return _cached_get_state_for_audio_prompt(tts_model, audio_conditioning, truncate)
