"""Thin entrypoint + backward-compatible exports for pocket-tts."""

from __future__ import annotations

# Typer app / CLI commands live in cli.py.
from pocket_tts.cli import app, cli_app, export_voice, generate, serve  # noqa: F401

# Backward-compatible server/global symbols used by older tests/scripts.
tts_model = None
default_voice_state = None

# Real ASGI app instance for backward compatibility (e.g. TestClient(main.web_app)).
from pocket_tts.server.app import create_app, web_app  # noqa: E402,F401

# Preserve old helper API used by tests.
from pocket_tts.server.app import generate_data_with_state as _generate_data_with_state_new  # noqa: E402


def generate_data_with_state(text_to_generate: str, model_state):
    model = tts_model
    if model is None:
        raise RuntimeError("no model loaded: `pocket-tts serve` loads it before serving requests")
    return _generate_data_with_state_new(text_to_generate, model_state, model)


# Keep compatibility with `from pocket_tts.main import TTSModel, uvicorn`.
from pocket_tts.models.tts_model import TTSModel  # noqa: E402,F401
import uvicorn  # noqa: E402,F401

__all__ = [
    "app",
    "cli_app",
    "serve",
    "generate",
    "export_voice",
    "web_app",
    "create_app",
    "tts_model",
    "default_voice_state",
    "generate_data_with_state",
    "TTSModel",
    "uvicorn",
]
