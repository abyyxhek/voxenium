"""FastAPI app for pocket-tts server."""

from __future__ import annotations

import io
import logging
import os
import tempfile
import threading
from collections.abc import AsyncIterator, Generator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Annotated, BinaryIO, cast

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pocket_tts.data.audio import stream_audio_chunks
from pocket_tts.default_parameters import get_default_text_for_language, get_default_voice_for_language
from pocket_tts.modules.stateful_module import ModelState
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.server.websocket import register_websocket_routes
from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

logger = logging.getLogger(__name__)


@dataclass
class ServerState:
    model: TTSModel | None = None
    default_voice_state: ModelState | None = None


def _compat_module():
    try:
        import pocket_tts.main as main

        return main
    except Exception:
        return None


def _resolve_model_from_compat(state: ServerState) -> ServerState:
    main = _compat_module()
    if main is not None:
        compat_model = getattr(main, "tts_model", None)
        if compat_model is not None:
            state.model = compat_model
        compat_default = getattr(main, "default_voice_state", None)
        if compat_default is not None:
            state.default_voice_state = compat_default
    return state


def _load_model_for_server() -> tuple[TTSModel, ModelState]:
    language = os.environ.get("POCKET_TTS_SERVE_LANGUAGE") or None
    config = os.environ.get("POCKET_TTS_SERVE_CONFIG") or None
    default_voice = os.environ.get("POCKET_TTS_SERVE_DEFAULT_VOICE") or None
    quantize = os.environ.get("POCKET_TTS_SERVE_QUANTIZE") == "1"
    model = TTSModel.load_model(language=language, config=config, quantize=quantize)
    if default_voice is None:
        default_voice = get_default_voice_for_language(language, config)
    voice_state = model.get_state_for_audio_prompt(default_voice)
    return model, voice_state


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    state = ServerState()
    # Tests and CLI compat can pre-populate module globals in pocket_tts.main.
    _resolve_model_from_compat(state)
    if state.model is None:
        # Normal server startup: load from env config / defaults.
        # Skip only when clearly running as a library/test context? Tests that need a
        # preloaded fake model set pocket_tts.main.tts_model before entering lifespan.
        load_default = os.environ.get("POCKET_TTS_SERVER_NO_AUTOLOAD") != "1"
        if load_default:
            try:
                state.model, state.default_voice_state = _load_model_for_server()
            except Exception:
                logger.exception("Failed to auto-load model during server startup")
    else:
        # Model supplied via compat globals (tests) - allow default voice injection too.
        _resolve_model_from_compat(state)
    app.state.server = state
    try:
        yield
    finally:
        app.state.server = ServerState()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Kyutai Pocket TTS API",
        description="Text-to-Speech generation API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "https://pod1-10007.internal.kyutai.org",
            "https://kyutai.org",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = Path(__file__).resolve().parents[1] / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    register_routes(app)
    register_websocket_routes(app)
    return app


def get_server_state(request: Request) -> ServerState:
    state = getattr(request.app.state, "server", None)
    if state is None:
        state = ServerState()
        request.app.state.server = state
    return _resolve_model_from_compat(state)


def get_tts_model(state: Annotated[ServerState, Depends(get_server_state)]) -> TTSModel:
    if state.model is None:
        raise HTTPException(status_code=503, detail="no model loaded: `pocket-tts serve` loads it before serving requests")
    return state.model


def get_default_voice_state(
    state: Annotated[ServerState, Depends(get_server_state)],
) -> ModelState | None:
    return state.default_voice_state


def write_to_queue(
    queue: Queue[bytes | None],
    text_to_generate: str,
    model_state: ModelState,
    model: TTSModel,
    stop: threading.Event,
):
    """Allows writing to the StreamingResponse as if it were a file."""

    class FileLikeToQueue(io.IOBase):
        def __init__(self, queue: Queue[bytes | None]):
            self.queue = queue

        def write(self, data: bytes):
            self.queue.put(data)

        def flush(self):
            pass

        def close(self):
            self.queue.put(None)

    audio_chunks = model.generate_audio_stream(
        model_state=model_state, text_to_generate=text_to_generate, stop=stop
    )
    stream_audio_chunks(
        cast(BinaryIO, FileLikeToQueue(queue)), audio_chunks, model.config.mimi.sample_rate
    )


def generate_data_with_state(
    text_to_generate: str, model_state: ModelState, model: TTSModel
) -> Generator[bytes, None, None]:
    queue: Queue[bytes | None] = Queue()
    stop = threading.Event()

    thread = threading.Thread(
        target=write_to_queue, args=(queue, text_to_generate, model_state, model, stop)
    )
    thread.start()

    try:
        while True:
            data = queue.get()
            if data is None:
                break
            yield data
    finally:
        stop.set()
        thread.join()


def register_routes(app: FastAPI) -> None:
    @app.get("/", response_class=HTMLResponse)
    async def root(model: Annotated[TTSModel, Depends(get_tts_model)]) -> str:
        """Serve the frontend."""
        static_path = Path(__file__).resolve().parents[1] / "static" / "index.html"
        content = static_path.read_text(encoding="utf-8")
        origin = str(model.origin)
        content = content.replace("DEFAULT_TEXT_PROMPT", get_default_text_for_language(origin))
        return content

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/voices")
    async def voices() -> list[str]:
        return sorted(_ORIGINS_OF_PREDEFINED_VOICES.keys())

    @app.post("/export-voice")
    async def export_voice_endpoint(
        model: Annotated[TTSModel, Depends(get_tts_model)],
        voice_wav: UploadFile = File(...),
    ) -> StreamingResponse:
        from pocket_tts.models.model_state import export_model_state

        suffix = Path(voice_wav.filename).suffix if voice_wav.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(voice_wav.file.read())
            temp_file.flush()
            temp_path = temp_file.name
        try:
            state = model.get_state_for_audio_prompt(Path(temp_path), truncate=True)
            out = tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors")
            out.close()
            export_model_state(state, out.name)
            with open(out.name, "rb") as f:
                data = f.read()
        finally:
            os.unlink(temp_path)
            try:
                os.unlink(out.name)
            except Exception:
                pass
        return StreamingResponse(
            iter([data]),
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=voice.safetensors"},
        )

    @app.post("/stream")
    def stream_endpoint(
        model: Annotated[TTSModel, Depends(get_tts_model)],
        default_state: Annotated[ModelState | None, Depends(get_default_voice_state)],
        text: str = Form(...),
        voice_url: str | None = Form(None),
        voice_wav: UploadFile | None = File(None),
    ) -> StreamingResponse:
        # Alias to /tts for compatibility if any clients used /stream.
        return _tts_impl(model, default_state, text, voice_url, voice_wav)

    @app.post("/tts")
    def text_to_speech(
        model: Annotated[TTSModel, Depends(get_tts_model)],
        default_state: Annotated[ModelState | None, Depends(get_default_voice_state)],
        text: str = Form(...),
        voice_url: str | None = Form(None),
        voice_wav: UploadFile | None = File(None),
    ) -> StreamingResponse:
        return _tts_impl(model, default_state, text, voice_url, voice_wav)


def _tts_impl(
    model: TTSModel,
    default_state: ModelState | None,
    text: str,
    voice_url: str | None,
    voice_wav: UploadFile | None,
) -> StreamingResponse:
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if voice_url is not None and voice_wav is not None:
        raise HTTPException(status_code=400, detail="Cannot provide both voice_url and voice_wav")

    if voice_url is not None:
        if not (
            voice_url.startswith("http://")
            or voice_url.startswith("https://")
            or voice_url.startswith("hf://")
            or voice_url in _ORIGINS_OF_PREDEFINED_VOICES
        ):
            raise HTTPException(
                status_code=400, detail="voice_url must start with http://, https://, or hf://"
            )
        model_state = model._cached_get_state_for_audio_prompt(voice_url)
        logger.warning("Using voice from URL: %s", voice_url)
    elif voice_wav is not None:
        suffix = Path(voice_wav.filename).suffix if voice_wav.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = voice_wav.file.read()
            temp_file.write(content)
            temp_file.flush()
            temp_file_path = temp_file.name
        try:
            model_state = model.get_state_for_audio_prompt(Path(temp_file_path), truncate=True)
        finally:
            os.unlink(temp_file_path)
    elif default_state is not None:
        model_state = default_state
    else:
        raise HTTPException(status_code=500, detail="The server has no default voice loaded.")

    return StreamingResponse(
        generate_data_with_state(text, model_state, model),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=generated_speech.wav",
            "Transfer-Encoding": "chunked",
        },
    )


# Compat export for scripts/tests expecting `web_app`.
web_app = create_app()
