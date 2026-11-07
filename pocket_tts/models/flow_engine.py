"""The pure FlowLM generation logic: latent sampling, KV-cache handling and the autoregressive loop."""

import logging
import os
import statistics
import threading
import torch

from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.modules.stateful_module import ModelState, increment_steps
from pocket_tts.models.streaming_decoder import LatentQueue
from pocket_tts.utils.utils import display_execution_time

logger = logging.getLogger(__name__)


def run_flow_lm(
    flow_lm: FlowLMModel,
    model_state: ModelState,
    text_tokens: torch.Tensor,
    backbone_input_latents: torch.Tensor,
    audio_conditioning: torch.Tensor,
    sampler_decode_steps: int,
    temp: float,
    noise_clamp: float | None,
    eos_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one FlowLM step, returning the decoded latent and the EOS prediction."""
    text_embeddings = flow_lm.conditioner(text_tokens)
    text_embeddings = torch.cat([text_embeddings, audio_conditioning], dim=1)

    output_embeddings, is_eos = flow_lm._sample_next_latent(
        backbone_input_latents,
        text_embeddings,
        model_state=model_state,
        sampler_decode_steps=sampler_decode_steps,
        temp=temp,
        noise_clamp=noise_clamp,
        eos_threshold=eos_threshold,
    )
    return output_embeddings[:, None, :], is_eos


def run_flow_lm_and_increment_step(
    flow_lm: FlowLMModel,
    model_state: ModelState,
    text_tokens: torch.Tensor | None = None,
    backbone_input_latents: torch.Tensor | None = None,
    audio_conditioning: torch.Tensor | None = None,
    sampler_decode_steps: int = 1,
    temp: float = 0.3,
    noise_clamp: float | None = None,
    eos_threshold: float = -4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one FlowLM step and increment the model state.

    First one is the backbone output, second one is the audio decoding output.
    """
    if text_tokens is None:
        text_tokens = torch.zeros((1, 0), dtype=torch.int64, device=flow_lm.device)
    if backbone_input_latents is None:
        backbone_input_latents = torch.empty(
            (1, 0, flow_lm.ldim), dtype=flow_lm.dtype, device=flow_lm.device
        )
    if audio_conditioning is None:
        audio_conditioning = torch.empty(
            (1, 0, flow_lm.dim), dtype=flow_lm.dtype, device=flow_lm.device
        )

    output = run_flow_lm(
        flow_lm,
        model_state,
        text_tokens,
        backbone_input_latents,
        audio_conditioning,
        sampler_decode_steps=sampler_decode_steps,
        temp=temp,
        noise_clamp=noise_clamp,
        eos_threshold=eos_threshold,
    )
    increment_by = (
        text_tokens.shape[1] + backbone_input_latents.shape[1] + audio_conditioning.shape[1]
    )
    increment_steps(flow_lm, model_state, increment=increment_by)
    return output


def expand_kv_cache(model_state: ModelState, sequence_length: int) -> None:
    """Expand KV cache back to full sequence_length for generation.

    When a model state is retrieved from cache with sliced KV caches,
    this method expands them back to the full size needed for generation.

    Args:
        model_state: The model state dict containing potentially sliced KV caches
        sequence_length: Target sequence length to expand caches to
    """
    for module_name, module_state in model_state.items():
        if "cache" in module_state:
            cache = module_state["cache"]
            # KV cache has shape [2, batch_size, current_length, num_kv_heads, dim_per_head]
            current_length = cache.shape[2]
            if current_length < sequence_length:
                # Create expanded cache filled with NaN for unused positions
                expanded_cache = torch.full(
                    (
                        cache.shape[0],
                        cache.shape[1],
                        sequence_length,
                        cache.shape[3],
                        cache.shape[4],
                    ),
                    float("NaN"),
                    device=cache.device,
                    dtype=cache.dtype,
                )
                # Copy existing data to the beginning
                expanded_cache[:, :, :current_length, :, :] = cache
                module_state["cache"] = expanded_cache


def flow_lm_current_end(model_state: ModelState) -> int:
    """Return the current sequence length tracked by the model state."""
    for module_state in model_state.values():
        offset = module_state.get("offset")
        if offset is not None:
            return int(offset.view(-1)[0].item())
    raise ValueError(
        "Could not find offset in model state, please open an issue "
        "at https://github.com/kyutai-labs/pocket-tts/issues"
    )


def autoregressive_generation(
    flow_lm: FlowLMModel,
    model_state: ModelState,
    max_gen_len: int,
    frames_after_eos: int,
    latents_queue: LatentQueue,
    stop: threading.Event,
    latents_out: list[torch.Tensor] | None = None,
    first_input: torch.Tensor | None = None,
    min_frames_before_eos: int = 6,
    sampler_decode_steps: int = 1,
    temp: float = 0.3,
    noise_clamp: float | None = None,
    eos_threshold: float = -4.0,
) -> None:
    """Run the autoregressive latent generation loop, queueing each latent for the decoder."""
    backbone_input = first_input
    if backbone_input is None:
        backbone_input = torch.full(
            (1, 1, flow_lm.ldim),
            fill_value=float("NaN"),
            device=next(iter(flow_lm.parameters())).device,
            dtype=flow_lm.dtype,
        )
    steps_times = []
    eos_step = None
    for generation_step in range(max_gen_len):
        if stop.is_set():
            break
        with display_execution_time("Generating latent", print_output=False) as timer:
            next_latent, is_eos = run_flow_lm_and_increment_step(
                flow_lm,
                model_state,
                backbone_input_latents=backbone_input,
                sampler_decode_steps=sampler_decode_steps,
                temp=temp,
                noise_clamp=noise_clamp,
                eos_threshold=eos_threshold,
            )
            if is_eos.item() and eos_step is None and generation_step >= min_frames_before_eos:
                eos_step = generation_step
            if eos_step is not None and generation_step >= eos_step + frames_after_eos:
                break

            # Add generated latent to queue for immediate decoding
            if latents_out is not None:
                latents_out.append(next_latent)
            latents_queue.put(next_latent)
            backbone_input = next_latent
        assert timer.elapsed_time_ms is not None
        steps_times.append(timer.elapsed_time_ms)
    else:
        if os.environ.get("KPOCKET_TTS_ERROR_WITHOUT_EOS", "0") == "1":
            raise RuntimeError("Generation reached maximum length without EOS!")
        logger.warning(
            "Maximum generation length reached without EOS, this very often indicates an error."
        )

    # Add sentinel value to signal end of generation
    latents_queue.put(None)
    if steps_times:
        logger.info("Average generation step time: %d ms", int(statistics.mean(steps_times)))
