"""CLI commands for pocket-tts."""
from __future__ import annotations

import logging
import os
import sys
from typing import Annotated

import typer
import uvicorn

from pocket_tts.data.audio import stream_audio_chunks
from pocket_tts.default_parameters import (
    DEFAULT_FRAMES_AFTER_EOS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_SAMPLER_DECODE_STEPS,
    MAX_TOKEN_PER_CHUNK,
    MAX_TOKEN_PER_CHUNK_TEACHER_FORCED,
    get_default_text_for_language,
    get_default_voice_for_language,
)
from pocket_tts.models.model_state import export_model_state
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.utils.logging_utils import enable_logging

logger = logging.getLogger(__name__)

cli_app = typer.Typer(
    help="Kyutai Pocket TTS - Text-to-Speech generation tool", pretty_exceptions_show_locals=False
)
# Backward-compatible alias requested by refactor spec.
app = cli_app


def _main_compat():
    try:
        import pocket_tts.main as main

        return main
    except Exception:
        return None


@cli_app.command()
def serve(
    host: Annotated[str, typer.Option(help="Host to bind to")] = "localhost",
    port: Annotated[int, typer.Option(help="Port to bind to")] = 8000,
    reload: Annotated[bool, typer.Option(help="Enable auto-reload")] = False,
    language: Annotated[
        str | None,
        typer.Option(
            help="Language for the TTS model. "
            "'english_2026-01', 'english_2026-04', 'english_2026-09', 'english_2026-09_2', 'english_2026-10', 'english_drifting_26-09', 'english_18m_2026-10', 'english_18m_refl_2026-10', 'english_7m_2026-10', 'english_7m_refl_2026-10', 'english_2m_2026-10', 'english_2m_refl_2026-10', 'english', 'french', 'french_24l', 'german', 'german_24l', 'portuguese', 'portuguese_24l', 'italian', 'italian_24l', 'spanish', 'spanish_24l', 'dutch', 'dutch_24l'."
            " Incompatible with the config argument. Default is 'english', which is the same model as 'english_2026-10'.",
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            help="Path to a model config .yaml file: a local path, an https:// URL, or an hf:// path. "
            "Incompatible with the language argument. If not provided, will use the default English model."
        ),
    ] = None,
    default_voice: Annotated[
        str | None,
        typer.Option(
            help="Voice used by requests that don't ask for one: a built-in voice name, "
            "a local path to an audio file or to a .safetensors voice, an https:// URL, "
            "or an hf:// path. Defaults to the built-in voice of the language.",
            show_default=False,
        ),
    ] = None,
    quantize: Annotated[
        bool, typer.Option(help="Apply int8 quantization to reduce memory usage")
    ] = False,
):
    """Start the FastAPI server."""
    main = _main_compat()
    model_cls = getattr(main, "TTSModel", TTSModel) if main is not None else TTSModel
    uv = getattr(main, "uvicorn", uvicorn) if main is not None else uvicorn

    model = model_cls.load_model(language=language, config=config, quantize=quantize)
    if default_voice is None:
        default_voice = get_default_voice_for_language(language, config)
    default_voice_state = model.get_state_for_audio_prompt(default_voice)

    if main is not None:
        main.tts_model = model
        main.default_voice_state = default_voice_state

    # Pass runtime config to the app factory for normal serving. When tests patch
    # uvicorn.run (or main.uvicorn.run), we still set compat globals above.
    os.environ["POCKET_TTS_SERVE_LANGUAGE"] = "" if language is None else language
    os.environ["POCKET_TTS_SERVE_CONFIG"] = "" if config is None else config
    os.environ["POCKET_TTS_SERVE_DEFAULT_VOICE"] = default_voice
    os.environ["POCKET_TTS_SERVE_QUANTIZE"] = "1" if quantize else "0"

    uv.run(
        "pocket_tts.server.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


@cli_app.command()
def generate(
    text: Annotated[str | None, typer.Option(help="Text to generate")] = None,
    voice: Annotated[
        str | None,
        typer.Option(
            help=(
                "Path to audio conditioning file (voice to clone). "
                "Defaults to a built-in voice chosen from the language: "
                "'giovanni' for italian, 'lola' for spanish, 'juergen' for german, "
                "'rafael' for portuguese, 'estelle' for french, 'alba' otherwise. "
                "With the config or checkpoint argument, defaults to alba's audio file, "
                "which any model can clone."
            ),
            show_default=False,
        ),
    ] = None,
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging output")] = False,
    language: Annotated[
        str | None,
        typer.Option(
            help=(
                "Language for the TTS model. "
                "'english_2026-01', 'english_2026-04', 'english_2026-09', 'english_2026-09_2', 'english_2026-10', 'english_drifting_26-09', 'english_18m_2026-10', 'english_18m_refl_2026-10', 'english_7m_2026-10', 'english_7m_refl_2026-10', 'english_2m_2026-10', 'english_2m_refl_2026-10', 'english', 'french', 'french_24l', 'german', 'german_24l', 'portuguese', 'portuguese_24l', 'italian', 'italian_24l', 'spanish', 'spanish_24l', 'dutch', 'dutch_24l'."
                " Incompatible with the config argument. Default is 'english', which is the same model as 'english_2026-10'. "
                "The '24l' variants are bigger models, "
                "not distilled yet and here only as preview. They're not the final "
                "models for those languages."
            ),
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            help="Path to a model config .yaml file: a local path, an https:// URL, or an hf:// path. "
            "Incompatible with the language argument. If not provided, will use the default English model."
        ),
    ] = None,
    checkpoint: Annotated[
        str | None,
        typer.Option(help="Training checkpoint (.pt) to load instead of the config's weights"),
    ] = None,
    sampler_decode_steps: Annotated[
        int, typer.Option(help="Number of generation steps")
    ] = DEFAULT_SAMPLER_DECODE_STEPS,
    lsd_decode_steps: Annotated[
        int | None, typer.Option(hidden=True, help="Deprecated: use --sampler-decode-steps")
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option(
            help="Temperature for generation. Defaults to the model's recommended "
            "value from its config (0.3)."
        ),
    ] = None,
    noise_clamp: Annotated[
        float | None, typer.Option(help="Noise clamp value")
    ] = DEFAULT_NOISE_CLAMP,
    eos_threshold: Annotated[
        float | None,
        typer.Option(help="EOS threshold. Defaults to the model's value from its config (-4.0)."),
    ] = None,
    frames_after_eos: Annotated[
        int | None, typer.Option(help="Number of frames to generate after EOS")
    ] = DEFAULT_FRAMES_AFTER_EOS,
    output_path: Annotated[
        str, typer.Option(help="Output path for generated audio")
    ] = "./tts_output.wav",
    device: Annotated[str, typer.Option(help="Device to use")] = "cpu",
    max_tokens: Annotated[
        int | None,
        typer.Option(
            help=f"Maximum number of tokens per chunk. Defaults to {MAX_TOKEN_PER_CHUNK}, "
            f"or {MAX_TOKEN_PER_CHUNK_TEACHER_FORCED} with --teacher-force-previous-chunk."
        ),
    ] = None,
    teacher_force_previous_chunk: Annotated[
        bool,
        typer.Option(
            help="Generate each chunk of a long text as the continuation of the previous one "
            "(its audio teacher-forced) instead of starting afresh from the voice prompt."
        ),
    ] = False,
    quantize: Annotated[
        bool, typer.Option(help="Apply int8 quantization to reduce memory usage")
    ] = False,
    speed: Annotated[float, typer.Option(help="Speech rate multiplier (0.5-2.0)")] = 1.0,
    normalize_text: Annotated[
        bool, typer.Option(help="Apply intelligent text normalization (numbers/currency/abbreviations)")
    ] = True,
):
    """Generate speech using Kyutai Pocket TTS."""
    if lsd_decode_steps is not None:
        logger.warning("--lsd-decode-steps is deprecated, use --sampler-decode-steps")
        sampler_decode_steps = lsd_decode_steps
    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        if text is None:
            text = get_default_text_for_language(language)
        if text == "-":
            text = sys.stdin.read()

        if not text.strip():
            logger.error("No input received from stdin.")
            raise typer.Exit(code=1)
        tts_model = TTSModel.load_model(
            language=language,
            config=config,
            temp=temperature,
            sampler_decode_steps=sampler_decode_steps,
            noise_clamp=noise_clamp,
            eos_threshold=eos_threshold,
            quantize=quantize,
            checkpoint=checkpoint,
        )
        tts_model.to(device)

        if voice is None:
            voice = get_default_voice_for_language(language, config, checkpoint)
        model_state_for_voice = tts_model.get_state_for_audio_prompt(voice)
        audio_chunks = tts_model.generate_audio_stream(
            model_state=model_state_for_voice,
            text_to_generate=text,
            frames_after_eos=frames_after_eos,
            max_tokens=max_tokens,
            teacher_force_previous_chunk=teacher_force_previous_chunk,
            speed=speed,
            normalize_text=normalize_text,
        )

        stream_audio_chunks(output_path, audio_chunks, tts_model.config.mimi.sample_rate)

        if output_path != "-":
            logger.info("Results written in %s", output_path)
        logger.info("-" * 20)
        logger.info(
            "If you want to try multiple voices and prompts quickly, try the `serve` command."
        )
        logger.info(
            "If you like Kyutai projects, comment, like, subscribe at https://x.com/kyutai_labs"
        )


@cli_app.command()
def export_voice(
    audio_path: Annotated[
        str, typer.Argument(help="Audio file or directory to convert and export")
    ],
    export_path: Annotated[str, typer.Argument(help="Output file or directory")],
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging output")] = False,
    language: Annotated[
        str | None,
        typer.Option(
            help=(
                "Language for the TTS model. "
                "'english_2026-01', 'english_2026-04', 'english_2026-09', 'english_2026-09_2', 'english_2026-10', 'english_drifting_26-09', 'english_18m_2026-10', 'english_18m_refl_2026-10', 'english_7m_2026-10', 'english_7m_refl_2026-10', 'english_2m_2026-10', 'english_2m_refl_2026-10', 'english', 'french', 'french_24l', 'german', 'german_24l', 'portuguese', 'portuguese_24l', 'italian', 'italian_24l', 'spanish', 'spanish_24l', 'dutch', 'dutch_24l'."
                " Incompatible with the config argument. Default is 'english', which is the same model as 'english_2026-10'. "
                "The '24l' variants are bigger models, "
                "not distilled yet and here only as preview."
            ),
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            help="Path to a model config .yaml file: a local path, an https:// URL, or an hf:// path. "
            "Incompatible with the language argument. If not provided, will use the default English model."
        ),
    ] = None,
):
    """Convert and save audio to .safetensors file"""
    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        tts_model = TTSModel.load_model(language=language, config=config)
        model_state = tts_model.get_state_for_audio_prompt(
            audio_conditioning=audio_path, truncate=True
        )
        export_model_state(model_state, export_path)


if __name__ == "__main__":
    cli_app()
