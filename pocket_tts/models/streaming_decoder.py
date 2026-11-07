"""Audio decoding worker: turns generated latents into audio chunks via the Mimi decoder."""

import logging
import queue
import time
from typing import Any

import torch

from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.models.mimi import MimiModel
from pocket_tts.models.prosody import apply_speed, clamp_speed
from pocket_tts.modules.stateful_module import increment_steps, init_states
from pocket_tts.utils.config import Config

logger = logging.getLogger(__name__)

# teacher_force_previous_chunk: the Mimi decoder of a chunk first runs over the last forced
# frames (about 1 s), so that its audio follows on from the previous chunk's.
_DECODER_WARMUP_FRAMES = 12

# Generated latents handed to the Mimi decoder thread; None closes the stream.
LatentQueue = queue.Queue[torch.Tensor | None]
# Decoder thread -> generator: ("chunk", audio), ("done", None) or ("error", exception).
ResultQueue = queue.Queue[tuple[str, Any]]


@torch.no_grad
def decode_audio_worker(
    mimi: MimiModel,
    flow_lm: FlowLMModel,
    config: Config,
    max_decoder_frames_per_call: int,
    latents_queue: LatentQueue,
    result_queue: ResultQueue,
    mimi_sequence_length: int,
    mimi_steps_per_latent: int,
    warmup_latents: torch.Tensor | None = None,
    speed: float = 1.0,
) -> None:
    """Worker thread function for decoding audio latents from queue with immediate streaming."""
    try:
        audio_chunks = []
        speed = clamp_speed(speed)
        mimi_state = init_states(mimi, batch_size=1, sequence_length=mimi_sequence_length)
        # A fresh Mimi decoder state puts a small step (about -41 dBFS) in its first samples, heard
        # as a click at the start of every chunk: fade the first 5 ms in. (The longer burst some
        # voices produced came from prompts ending on speech, handled by end_on_pause.)
        fade_in: torch.Tensor | None = torch.linspace(0, 1, int(config.mimi.sample_rate) // 200)
        if warmup_latents is not None:
            # Continuation of the previous chunk: run the decoder over its last frames (audio
            # dropped) so the new chunk's audio follows on without a fresh-state transient.
            mimi.decode_from_latent(warmup_latents * flow_lm.emb_std + flow_lm.emb_mean, mimi_state)
            increment_steps(
                mimi, mimi_state, increment=mimi_steps_per_latent * warmup_latents.shape[1]
            )
            fade_in = None
        while True:
            latent = latents_queue.get()
            if latent is None:
                break
            # Decode every latent frame the generator has queued in one call. The first frame
            # never waits. max_decoder_frames_per_call=1 is frame-by-frame decoding.
            latents = [latent]
            finished = False
            while not finished and (
                max_decoder_frames_per_call <= 0 or len(latents) < max_decoder_frames_per_call
            ):
                try:
                    nxt = latents_queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    finished = True
                else:
                    latents.append(nxt)
            mimi_decoding_input = torch.cat(latents, dim=1) * flow_lm.emb_std + flow_lm.emb_mean

            t = time.monotonic()
            audio_frame = mimi.decode_from_latent(mimi_decoding_input, mimi_state)
            if fade_in is not None:
                n = min(fade_in.numel(), audio_frame.shape[-1])
                audio_frame[..., :n] *= fade_in[:n].to(audio_frame)
                fade_in = None
            if speed != 1.0:
                audio_frame = apply_speed(audio_frame, speed)
            increment_steps(mimi, mimi_state, increment=mimi_steps_per_latent * len(latents))
            audio_frame_duration = audio_frame.shape[2] / config.mimi.sample_rate
            logger.debug(
                " " * 30 + "Decoded %d ms of audio (%d frames) with mimi in %d ms",
                int(audio_frame_duration * 1000),
                len(latents),
                int((time.monotonic() - t) * 1000),
            )
            audio_chunks.append(audio_frame)

            result_queue.put(("chunk", audio_frame))

            for _ in latents:
                latents_queue.task_done()
            if finished:
                break

        # Signal completion
        result_queue.put(("done", None))

    except Exception as e:
        # Put error in result queue
        result_queue.put(("error", e))
