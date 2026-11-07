"""The public TTSModel facade over the focused modules of ``pocket_tts.models``."""

import copy
import logging
import math
import queue
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from typing_extensions import Self

from pocket_tts.default_parameters import (
    DEFAULT_LANGUAGE,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_SAMPLER_DECODE_STEPS,
    MAX_TOKEN_PER_CHUNK,
    MAX_TOKEN_PER_CHUNK_TEACHER_FORCED,
)
from pocket_tts.models.flow_engine import (
    autoregressive_generation,
    expand_kv_cache,
    flow_lm_current_end,
    run_flow_lm,
    run_flow_lm_and_increment_step,
)
from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.models.mimi import MimiModel
from pocket_tts.models.model_loader import (
    _from_pydantic_config,
    _from_pydantic_config_with_weights,
    load_training_checkpoint,
    quantize_model,
)
from pocket_tts.models.model_state import _is_safetensors_source  # noqa: F401
from pocket_tts.models.streaming_decoder import _DECODER_WARMUP_FRAMES, decode_audio_worker
from pocket_tts.models.text_chunking import prepare_text_prompt, split_into_best_sentences
from pocket_tts.models.voice_processor import (
    VOICE_CLONING_UNSUPPORTED,  # noqa: F401
    _cached_get_state_for_audio_prompt,
    decode_and_dump,
    encode_prompt,
    get_state_for_audio_prompt as get_voice_state,
)
from pocket_tts.modules.stateful_module import ModelState
from pocket_tts.utils.config import CONFIGS_DIR, Config, load_config
from pocket_tts.utils.utils import DEBUG_MIMI, display_execution_time


torch.set_num_threads(1)
logger = logging.getLogger(__name__)

# Generated latents handed to the Mimi decoder thread; None closes the stream.
LatentQueue = queue.Queue[torch.Tensor | None]
# Decoder thread -> generator: ("chunk", audio), ("done", None) or ("error", exception).
ResultQueue = queue.Queue[tuple[str, Any]]


class TTSModel(nn.Module):
    flow_lm: FlowLMModel
    mimi: MimiModel

    _TOKENS_PER_SECOND_ESTIMATE = 3.0
    _GEN_SECONDS_PADDING = 2.0
    # EOS is ignored on the first frames: before speech starts, the EOS logit of some voices can
    # cross the threshold, and a short text then ends before the word is spoken.
    _MIN_FRAMES_BEFORE_EOS = 6

    def __init__(
        self,
        flow_lm: FlowLMModel,
        temp: float,
        sampler_decode_steps: int,
        noise_clamp: float | None,
        eos_threshold: float,
        config: Config,
        origin: Path | None = None,
        pad_with_spaces_for_short_inputs: bool = False,
        model_recommended_frames_after_eos: int | None = None,
        remove_semicolons: bool = False,
        append_terminal_punctuation: bool = True,
        capitalize_first_letter: bool = True,
        replace_characters: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.flow_lm = flow_lm
        # 0 = decode every queued frame in one call; 1 = one frame per call
        self.max_decoder_frames_per_call = 0
        self.temp = temp
        self.sampler_decode_steps = sampler_decode_steps
        self.noise_clamp = noise_clamp
        self.eos_threshold = eos_threshold
        self.config = config
        self.has_voice_cloning = True
        self.origin = origin
        self.pad_with_spaces_for_short_inputs: bool = pad_with_spaces_for_short_inputs
        self.model_recommended_frames_after_eos = model_recommended_frames_after_eos
        self.remove_semicolons = remove_semicolons
        self.append_terminal_punctuation = append_terminal_punctuation
        self.capitalize_first_letter = capitalize_first_letter
        self.replace_characters = replace_characters or {}

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def sample_rate(self) -> int:
        return self.config.mimi.sample_rate

    @classmethod
    def _from_pydantic_config(
        cls,
        config: Config,
        temp: float,
        sampler_decode_steps: int,
        noise_clamp: float | None,
        eos_threshold: float,
        origin: Path | None,
    ) -> Self:
        return cast(
            Self,
            _from_pydantic_config(
                config, temp, sampler_decode_steps, noise_clamp, eos_threshold, origin=origin
            ),
        )

    has_custom_weights: bool = False

    def load_training_checkpoint(self, path: str | Path, use_ema: bool = True) -> None:
        """Load weights from a training checkpoint (.pt) into a config-built model."""
        load_training_checkpoint(self, path, use_ema)

    @classmethod
    def _from_pydantic_config_with_weights(
        cls,
        config: Config,
        temp: float,
        sampler_decode_steps: int,
        noise_clamp: float | None,
        eos_threshold: float,
        origin: Path | None = None,
    ) -> Self:
        return cast(
            Self,
            _from_pydantic_config_with_weights(
                config, temp, sampler_decode_steps, noise_clamp, eos_threshold, origin=origin
            ),
        )

    @classmethod
    def load_model(
        cls,
        language: str | None = None,
        config: str | Path | None = None,
        temp: float | None = None,
        sampler_decode_steps: int = DEFAULT_SAMPLER_DECODE_STEPS,
        noise_clamp: float | None = DEFAULT_NOISE_CLAMP,
        eos_threshold: float | None = None,
        quantize: bool = False,
        checkpoint: str | Path | None = None,
        lsd_decode_steps: int | None = None,
    ) -> Self:
        """Load a pre-trained TTS model with specified configuration.

        This class method loads a complete TTS model including the flow language model
        and Mimi compression model from pre-trained weights. The model is initialized
        with the specified generation parameters and ready for inference.

        Args:
            language: Optional language identifier to select a predefined config. Incompatible with
                the `config` argument. Available options
                are `"english_2026-01"`, `"english_2026-04"`, `"english_2026-09"`, `"english_2026-09_2"`, `"english_2026-10"`, `"english_drifting_26-09"`, `"english_18m_2026-10"`, `"english_18m_refl_2026-10"`, `"english_7m_2026-10"`, `"english_7m_refl_2026-10"`, `"english_2m_2026-10"`, `"english_2m_refl_2026-10"`, `"english_7m_2026-10"`, `"english_7m_refl_2026-10"`, `"english_2m_2026-10"`, `"english_2m_refl_2026-10"`, `"english"`, `"french"`, `"french_24l"`, `"german"`, `"german_24l"`, `"portuguese"`, `"portuguese_24l"`, `"italian"`, `"italian_24l"`, `"spanish"`, `"spanish_24l"`, `"dutch"`, `"dutch_24l"`.
                If neither `config` nor `language` is provided, defaults to `"english", which is the same model as 'english_2026-10'`.
            config: A path to a custom YAML config file: a local path (e.g., `"C://pocket_tts/pocket_tts_config.yaml"`,
                an `https://` URL, or an `hf://` path (e.g. `"hf://<repo_id>/<path>[@revision]"`).
            temp: Sampling temperature for generation. Higher values produce more
                diverse but potentially lower quality output. If None, defaults to
                the model's recommended value from its config file
                (``default_temperature``, 0.3).
            sampler_decode_steps: Number of steps for Lagrangian Self Distillation
                decoding. More steps can improve quality but increase computation.
            noise_clamp: Maximum value for noise sampling. If None, no clamping
                is applied. Helps prevent extreme values in generation.
            eos_threshold: Threshold for end-of-sequence detection. Higher values
                make the model more likely to continue generating. If None, defaults to
                the model's value from its config file (``default_eos_threshold``, -4.0).
            checkpoint: Optional training checkpoint (.pt) to load instead of the
                config's safetensors weights, so any step of a run can be played
                without exporting first. EMA weights are used when the checkpoint
                carries them.
            quantize: If True, apply dynamic int8 quantization to the transformer's
                attention and FFN layers. Reduces runtime memory by ~48% and improves
                inference speed by ~27% on x86 (FBGEMM).
                No measurable impact on speech quality (WER unchanged).
                For optimized performance, install torchao: ``pip install pocket-tts[quantize]``

        Returns:
            TTSModel: Fully initialized model with loaded weights on cpu, ready for
                text-to-speech generation.

        Raises:
            FileNotFoundError: If the specified config file or model weights
                are not found.
            ValueError: If the configuration is invalid or incompatible.

        Example:
            ```python
            from pocket_tts import TTSModel

            # Load with default settings
            model = TTSModel.load_model()

            # Load with int8 quantization
            model = TTSModel.load_model(quantize=True)
            ```
        """
        if config is not None and language is not None:
            raise ValueError(
                "Cannot specify both config and language, please choose one or the other."
            )
        if config is None and language is None:
            language = DEFAULT_LANGUAGE
        if language is not None:
            config = CONFIGS_DIR / f"{language}.yaml"
        if lsd_decode_steps is not None:
            logger.warning("lsd_decode_steps is deprecated, use sampler_decode_steps")
            sampler_decode_steps = lsd_decode_steps

        config_path = str(config)
        # hf:// paths may carry a "@revision" suffix, strip it before checking the extension.
        suffix_source = (
            config_path.split("@")[0] if config_path.startswith("hf://") else config_path
        )
        if Path(suffix_source).suffix not in (".yaml", ".yml"):
            raise ValueError("Config should be a path to a YAML file ending with .yaml")
        model_config = load_config(config_path)
        if temp is None:
            temp = model_config.default_temperature
        if eos_threshold is None:
            eos_threshold = model_config.default_eos_threshold
        logger.info(f"Loading model from config at {config_path}...")

        origin = Path(config_path)
        if checkpoint is not None:
            tts_model = cls._from_pydantic_config(
                model_config, temp, sampler_decode_steps, noise_clamp, eos_threshold, origin=origin
            )
            tts_model.load_training_checkpoint(checkpoint)
        else:
            tts_model = cls._from_pydantic_config_with_weights(
                model_config, temp, sampler_decode_steps, noise_clamp, eos_threshold, origin=origin
            )

        if quantize:
            quantize_model(tts_model)

        return tts_model

    def _run_flow_lm_and_increment_step(
        self,
        model_state: ModelState,
        text_tokens: torch.Tensor | None = None,
        backbone_input_latents: torch.Tensor | None = None,
        audio_conditioning: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """First one is the backbone output, second one is the audio decoding output."""
        return run_flow_lm_and_increment_step(
            self.flow_lm,
            model_state,
            text_tokens=text_tokens,
            backbone_input_latents=backbone_input_latents,
            audio_conditioning=audio_conditioning,
            sampler_decode_steps=self.sampler_decode_steps,
            temp=self.temp,
            noise_clamp=self.noise_clamp,
            eos_threshold=self.eos_threshold,
        )

    def _run_flow_lm(
        self,
        model_state: ModelState,
        text_tokens: torch.Tensor,
        backbone_input_latents: torch.Tensor,
        audio_conditioning: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return run_flow_lm(
            self.flow_lm,
            model_state,
            text_tokens,
            backbone_input_latents,
            audio_conditioning,
            sampler_decode_steps=self.sampler_decode_steps,
            temp=self.temp,
            noise_clamp=self.noise_clamp,
            eos_threshold=self.eos_threshold,
        )

    def _decode_and_dump(self, encoded: torch.Tensor, filename: str) -> None:
        decode_and_dump(self.mimi, encoded, filename)

    def _encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        return encode_prompt(self.flow_lm, self.mimi, audio, debug_mimi=DEBUG_MIMI)

    def _expand_kv_cache(self, model_state: ModelState, sequence_length: int) -> None:
        """Expand KV cache back to full sequence_length for generation.

        When a model state is retrieved from cache with sliced KV caches,
        this method expands them back to the full size needed for generation.

        Args:
            model_state: The model state dict containing potentially sliced KV caches
            sequence_length: Target sequence length to expand caches to
        """
        expand_kv_cache(model_state, sequence_length)

    def _flow_lm_current_end(self, model_state: ModelState) -> int:
        return flow_lm_current_end(model_state)

    def _decode_audio_worker(
        self,
        latents_queue: LatentQueue,
        result_queue: ResultQueue,
        mimi_sequence_length: int,
        mimi_steps_per_latent: int,
        warmup_latents: torch.Tensor | None = None,
    ) -> None:
        """Worker thread function for decoding audio latents from queue with immediate streaming."""
        decode_audio_worker(
            mimi=self.mimi,
            flow_lm=self.flow_lm,
            config=self.config,
            max_decoder_frames_per_call=self.max_decoder_frames_per_call,
            latents_queue=latents_queue,
            result_queue=result_queue,
            mimi_sequence_length=mimi_sequence_length,
            mimi_steps_per_latent=mimi_steps_per_latent,
            warmup_latents=warmup_latents,
        )

    @torch.no_grad
    def generate_audio(
        self,
        model_state: ModelState,
        text_to_generate: str,
        max_tokens: int | None = None,
        frames_after_eos: int | None = None,
        copy_state: bool = True,
        teacher_force_previous_chunk: bool = False,
    ) -> torch.Tensor:
        """Generate complete audio tensor from text input.

        This method generates the full audio output for the given text prompt
        and returns it as a single tensor. It internally uses the streaming
        generation method but collects all chunks before returning.

        This method is NOT thread-safe; separate model instances should be used
        for concurrent generation.

        Args:
            model_state: Model state dictionary containing hidden states and
                positional information. Can be obtained from get_state_for_audio_prompt()
                or init_states(). The state may be modified during generation.
            text_to_generate: Input text to convert to speech. The text will be
                automatically formatted (capitalization, punctuation) for optimal
                generation quality.
            frames_after_eos: Number of additional frames to generate after
                detecting end-of-sequence. If None, automatically determined
                based on text length (1-3 frames).
            copy_state: Whether to create a deep copy of the model state before
                generation. If True, preserves the original state for reuse.
                If False, modifies the input state in-place. Defaults to True.
            max_tokens: Long texts are generated in chunks of at most this many tokens.
                Defaults to 50, or 25 with `teacher_force_previous_chunk`.
            teacher_force_previous_chunk: When True, each chunk after the first is generated
                from the voice prompt with the text of the previous chunk and of this one, the
                previous chunk's generated latents being teacher-forced as the first frames, so
                that the new chunk continues the previous one instead of starting afresh. Costs
                one prefill of the previous chunk's frames and text per chunk. Forces
                `copy_state=True`. Defaults to False.

        Returns:
            torch.Tensor: Generated audio tensor with shape [channels, samples]
                at the model's sample rate (typically 24kHz). The audio is
                normalized and ready for playback or saving.
                You can get the sample rate from the `sample_rate` attribute.

        Raises:
            ValueError: If text_to_generate is empty or invalid.
            RuntimeError: If generation fails due to model errors.

        Example:
            ```python
            from pocket_tts import TTSModel

            model = TTSModel.load_model()

            voice_state = model.get_state_for_audio_prompt("hf://kyutai/tts-voices/alba-mackenna/casual.wav")

            # Generate audio
            audio = model.generate_audio(voice_state, "Hello world!", frames_after_eos=2, copy_state=True)

            print(f"Generated audio shape: {audio.shape}")
            print(f"Audio duration: {audio.shape[-1] / model.sample_rate:.2f} seconds")
            ```
        """
        audio_chunks = []
        for chunk in self.generate_audio_stream(
            model_state=model_state,
            text_to_generate=text_to_generate,
            frames_after_eos=frames_after_eos,
            copy_state=copy_state,
            max_tokens=max_tokens,
            teacher_force_previous_chunk=teacher_force_previous_chunk,
        ):
            audio_chunks.append(chunk)
        return torch.cat(audio_chunks, dim=0)

    @torch.no_grad
    def generate_audio_stream(
        self,
        model_state: ModelState,
        text_to_generate: str,
        max_tokens: int | None = None,
        frames_after_eos: int | None = None,
        copy_state: bool = True,
        stop: threading.Event | None = None,
        teacher_force_previous_chunk: bool = False,
    ) -> Iterator[torch.Tensor]:
        """Generate audio streaming chunks from text input.

        This method generates audio from text and yields chunks as they become
        available, enabling real-time playback or processing. It uses multithreading
        to parallelize generation and decoding for optimal performance.
        This method is NOT thread-safe; separate model instances should be used
        for concurrent generation.

        Args:
            model_state: Model state dictionary containing hidden states and
                positional information. Can be obtained from get_state_for_audio_prompt()
                or init_states(). The state may be modified during generation.
            text_to_generate: Input text to convert to speech. The text will be
                automatically formatted (capitalization, punctuation) for optimal
                generation quality.
            frames_after_eos: Number of additional frames to generate after
                detecting end-of-sequence. If None, automatically determined
                based on text length (1-3 frames). Defaults to None.
            copy_state: Whether to create a deep copy of the model state before
                generation. If True, preserves the original state for reuse.
                If False, modifies the input state in-place. Defaults to True.
            max_tokens: Long texts are generated in chunks of at most this many tokens.
                Defaults to 50, or 25 with `teacher_force_previous_chunk`.
            teacher_force_previous_chunk: When True, each chunk after the first is generated
                from the voice prompt with the text of the previous chunk and of this one, the
                previous chunk's generated latents being teacher-forced as the first frames, so
                that the new chunk continues the previous one instead of starting afresh. Costs
                one prefill of the previous chunk's frames and text per chunk. Forces
                `copy_state=True`. Defaults to False.
            stop: Optional event for cancelling the generation, for instance
                when the user interrupts the playback. Once set, no new frames
                are generated and the stream ends early.

        Yields:
            torch.Tensor: Audio chunks with shape [samples] at the model's
                sample rate (typically 24kHz). Chunks are yielded as soon as
                they are decoded, enabling real-time streaming.

        Raises:
            ValueError: If text_to_generate is empty or invalid.
            RuntimeError: If generation fails due to model errors or threading issues.

        Example:
            ```python
            from pocket_tts import TTSModel

            model = TTSModel.load_model()

            voice_state = model.get_state_for_audio_prompt("hf://kyutai/tts-voices/alba-mackenna/casual.wav")
            # Stream generation
            for chunk in model.generate_audio_stream(voice_state, "Long text content..."):
                # Process each chunk as it's generated
                print(f"Generated chunk: {chunk.shape[0]} samples")
                # Could save chunks to file or play in real-time
            ```

        Note:
            This method uses multithreading to parallelize latent generation
            and audio decoding. Generation performance is logged including
            real-time factor (RTF) metrics.
        """
        if frames_after_eos is None:
            frames_after_eos = self.model_recommended_frames_after_eos
        if stop is None:
            stop = threading.Event()

        if max_tokens is None:
            max_tokens = (
                MAX_TOKEN_PER_CHUNK_TEACHER_FORCED
                if teacher_force_previous_chunk
                else MAX_TOKEN_PER_CHUNK
            )
        # Long texts are generated chunk by chunk. By default every chunk starts afresh from the
        # voice prompt; with teacher_force_previous_chunk it continues the previous one.
        chunks = split_into_best_sentences(
            self.flow_lm.conditioner.tokenizer,
            text_to_generate,
            max_tokens,
            self.pad_with_spaces_for_short_inputs,
            remove_semicolons=self.remove_semicolons,
            append_terminal_punctuation=self.append_terminal_punctuation,
            capitalize_first_letter=self.capitalize_first_letter,
            replace_characters=self.replace_characters,
        )

        prev_chunk: str | None = None
        prev_latents: list[torch.Tensor] = []
        for chunk in chunks:
            if stop.is_set():
                break
            forced, text_chunk = None, chunk
            if teacher_force_previous_chunk and prev_latents:
                forced = torch.cat(prev_latents, dim=1)
                text_chunk = f"{prev_chunk} {chunk}"
            prev_chunk, prev_latents = chunk, []
            text_to_generate, _ = prepare_text_prompt(
                text_chunk,
                self.pad_with_spaces_for_short_inputs,
                self.remove_semicolons,
                self.append_terminal_punctuation,
                self.capitalize_first_letter,
                self.replace_characters,
            )
            # The EOS delay depends on the new chunk only.
            _, frames_after_eos_guess = prepare_text_prompt(
                chunk,
                self.pad_with_spaces_for_short_inputs,
                self.remove_semicolons,
                self.append_terminal_punctuation,
                self.capitalize_first_letter,
                self.replace_characters,
            )
            frames_after_eos_guess += 2
            effective_frames = (
                frames_after_eos if frames_after_eos is not None else frames_after_eos_guess
            )
            yield from self._generate_audio_stream_short_text(
                model_state=model_state,
                text_to_generate=text_to_generate,
                frames_after_eos=effective_frames,
                copy_state=copy_state or teacher_force_previous_chunk,
                stop=stop,
                latents_out=prev_latents if teacher_force_previous_chunk else None,
                forced_latents=forced,
            )

    @torch.no_grad
    def _generate_audio_stream_short_text(
        self,
        model_state: ModelState,
        text_to_generate: str,
        frames_after_eos: int,
        copy_state: bool,
        stop: threading.Event,
        latents_out: list[torch.Tensor] | None = None,
        forced_latents: torch.Tensor | None = None,
    ) -> Iterator[torch.Tensor]:
        if copy_state:
            model_state = copy.deepcopy(model_state)

        prepared = self.flow_lm.conditioner.prepare(text_to_generate)
        token_count = prepared.shape[1]
        # With forced frames the text also holds the previous chunk's: a looser upper bound.
        max_gen_len = self._estimate_max_gen_len(token_count)
        mimi_steps_per_latent = int(self.mimi.encoder_frame_rate / self.mimi.frame_rate)
        warmup = None
        if forced_latents is not None:
            warmup = forced_latents[:, -_DECODER_WARMUP_FRAMES:]
        n_warmup = 0 if warmup is None else warmup.shape[1]
        mimi_sequence_length = (max_gen_len + n_warmup) * mimi_steps_per_latent

        # Set up multithreaded generation and decoding
        latents_queue: LatentQueue = queue.Queue()
        result_queue: ResultQueue = queue.Queue()

        # Start decoder worker thread
        decoder_thread = threading.Thread(
            target=self._decode_audio_worker,
            args=(latents_queue, result_queue, mimi_sequence_length, mimi_steps_per_latent, warmup),
            daemon=True,
        )
        logger.info("starting timer now!")
        t_generating = time.monotonic()
        decoder_thread.start()

        # Generate latents and add them to queue (decoder processes them in parallel)
        self._generate(
            model_state=model_state,
            prepared=prepared,
            max_gen_len=max_gen_len,
            frames_after_eos=frames_after_eos,
            latents_queue=latents_queue,
            result_queue=result_queue,
            stop=stop,
            latents_out=latents_out,
            forced_latents=forced_latents,
        )

        # Stream audio chunks as they become available
        total_generated_samples = 0
        while True:
            result = result_queue.get()
            if result[0] == "chunk":
                # Audio chunk available immediately for streaming/playback
                audio_chunk = result[1]
                total_generated_samples += audio_chunk.shape[-1]
                yield audio_chunk[0, 0]  # Remove batch, channel
            elif result[0] == "done":
                # Generation complete
                break
            elif result[0] == "error":
                # Wait for decoder thread to finish cleanly before propagating error
                with display_execution_time("Waiting for mimi decoder to finish"):
                    decoder_thread.join()
                # Propagate error
                raise result[1]

        # Wait for decoder thread to finish cleanly
        with display_execution_time("Waiting for mimi decoder to finish"):
            decoder_thread.join()

        # Print timing information
        duration_generated_audio = int(
            total_generated_samples * 1000 / self.config.mimi.sample_rate
        )
        generation_time = int((time.monotonic() - t_generating) * 1000)
        real_time_factor = duration_generated_audio / generation_time

        logger.info(
            "Generated: %d ms of audio in %d ms so %.2fx faster than real-time",
            duration_generated_audio,
            generation_time,
            real_time_factor,
        )

    @torch.no_grad
    def _generate(
        self,
        model_state: ModelState,
        prepared: torch.Tensor,
        max_gen_len: int,
        frames_after_eos: int,
        latents_queue: LatentQueue,
        result_queue: ResultQueue,
        stop: threading.Event,
        latents_out: list[torch.Tensor] | None = None,
        forced_latents: torch.Tensor | None = None,
    ) -> None:
        token_count = prepared.shape[1]
        n_forced = 0 if forced_latents is None else forced_latents.shape[1]
        current_end = self._flow_lm_current_end(model_state)
        required_len = current_end + token_count + n_forced + max_gen_len
        self._expand_kv_cache(model_state, sequence_length=required_len)

        first_input = None
        with display_execution_time("Prompting text"):
            if forced_latents is None or n_forced == 0:
                self._run_flow_lm_and_increment_step(model_state=model_state, text_tokens=prepared)
            else:
                # Teacher forcing: BOS then every forced frame but the last, which is the input
                # of the first generated step. The outputs of this prefill, EOS included, are
                # discarded: EOS can only fire on new frames.
                bos = torch.full_like(forced_latents[:, :1], float("NaN"))
                self._run_flow_lm_and_increment_step(
                    model_state=model_state,
                    text_tokens=prepared,
                    backbone_input_latents=torch.cat([bos, forced_latents[:, :-1]], dim=1),
                )
                first_input = forced_latents[:, -1:]

        def run_generation() -> None:
            try:
                self._autoregressive_generation(
                    model_state,
                    max_gen_len,
                    frames_after_eos,
                    latents_queue,
                    stop,
                    latents_out,
                    first_input,
                )
            except Exception as e:
                logger.error(f"Error in autoregressive generation: {e}")
                # Report the generation error before stopping the decoder. Otherwise
                # the decoder can publish "done" first and hide the exception.
                if result_queue is not None:
                    result_queue.put(("error", e))
                # Signal decoder to stop by putting None (completion sentinel)
                if latents_queue is not None:
                    latents_queue.put(None)

        generation_thread = threading.Thread(target=run_generation, daemon=True)
        generation_thread.start()

    @torch.no_grad
    def _autoregressive_generation(
        self,
        model_state: ModelState,
        max_gen_len: int,
        frames_after_eos: int,
        latents_queue: LatentQueue,
        stop: threading.Event,
        latents_out: list[torch.Tensor] | None = None,
        first_input: torch.Tensor | None = None,
    ) -> None:
        autoregressive_generation(
            self.flow_lm,
            model_state,
            max_gen_len,
            frames_after_eos,
            latents_queue,
            stop,
            latents_out=latents_out,
            first_input=first_input,
            min_frames_before_eos=self._MIN_FRAMES_BEFORE_EOS,
            sampler_decode_steps=self.sampler_decode_steps,
            temp=self.temp,
            noise_clamp=self.noise_clamp,
            eos_threshold=self.eos_threshold,
        )

    def _cached_get_state_for_audio_prompt(
        self, audio_conditioning: Path | str | torch.Tensor, truncate: bool = False
    ) -> ModelState:
        """Memoize the voice states: encoding an audio prompt is the expensive part."""
        return _cached_get_state_for_audio_prompt(self, audio_conditioning, truncate)

    @torch.no_grad
    def get_state_for_audio_prompt(
        self, audio_conditioning: Path | str | torch.Tensor, truncate: bool = False
    ) -> ModelState:
        """Create model state conditioned on audio prompt for continuation.

        This method processes an audio prompt and creates a model state that
        captures the acoustic characteristics (speaker voice, style, prosody)
        for use in subsequent text-to-speech generation. The resulting state
        enables voice cloning and audio continuation with speaker consistency.

        Args:
            audio_conditioning: Audio prompt to condition (or .safetensors to load). Can be:
                - Path: Local file path to audio file (or .safetensors)
                - str: URL to download audio file (or .safetensors) from
                - torch.Tensor: Pre-loaded audio tensor with shape [channels, samples]
            truncate: Whether to truncate long audio prompts to 30 seconds.
                Helps prevent memory issues with very long inputs. Defaults to False.

        Returns:
            dict: Model state dictionary containing hidden states and positional
                information conditioned on the audio prompt. This state can be
                passed to `generate_audio()` or `generate_audio_stream()` for
                voice-consistent generation.

        Raises:
            FileNotFoundError: If audio file path doesn't exist.
            ValueError: If audio tensor is invalid or empty.
            RuntimeError: If audio processing or encoding fails.

        Example:
            ```python
            from pocket_tts import TTSModel

            model = TTSModel.load_model()
            # From HuggingFace URL
            voice_state = model.get_state_for_audio_prompt("hf://kyutai/tts-voices/alba-mackenna/casual.wav")

            # From local file
            voice_state = model.get_state_for_audio_prompt("./my_voice.wav")

            # Reload state from a .safetensors file (much faster than extracting from an audio file)
            voice_state = model.get_state_for_audio_prompt("./my_voices.safetensors")

            # From HTTP URL
            voice_state = model.get_state_for_audio_prompt(
                "https://huggingface.co/kyutai/tts-voices/resolve"
                "/main/expresso/ex01-ex02_default_001_channel1_168s.wav"
            )
            ```

        Note:
            - Audio is automatically resampled to the model's sample rate (24kHz)
            - The audio is encoded using the Mimi compression model and projected
              to the flow model's latent space
            - Processing time is logged for performance monitoring
            - The state preserves speaker characteristics for voice cloning
        """
        return get_voice_state(
            self.flow_lm,
            self.mimi,
            audio_conditioning,
            device=self.device,
            has_voice_cloning=self.has_voice_cloning,
            has_custom_weights=self.has_custom_weights,
            origin=self.origin,
            config=self.config,
            truncate=truncate,
            sampler_decode_steps=self.sampler_decode_steps,
            temp=self.temp,
            noise_clamp=self.noise_clamp,
            eos_threshold=self.eos_threshold,
        )

    def _estimate_max_gen_len(self, token_count: int) -> int:
        gen_len_sec = token_count / self._TOKENS_PER_SECOND_ESTIMATE + self._GEN_SECONDS_PADDING
        frame_rate = self.config.mimi.frame_rate
        return math.ceil(gen_len_sec * frame_rate)
