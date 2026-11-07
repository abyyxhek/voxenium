"""WebSocket real-time streaming endpoint."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES


def _tensor_to_pcm16_bytes(audio_chunk: Any) -> bytes:
    # audio_chunk is expected to be 1D float tensor in [-1, 1] at 24kHz.
    import torch

    if isinstance(audio_chunk, torch.Tensor):
        t = audio_chunk.detach()
        if t.is_cuda:
            t = t.cpu()
        pcm = (t.float().clamp(-1, 1) * 32767.0).to(torch.int16)
        return pcm.numpy().tobytes(order="C")
    # Fallback for numpy-like arrays.
    import numpy as np

    arr = np.asarray(audio_chunk, dtype=np.float32)
    pcm = np.clip(arr, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    return pcm.tobytes()


def register_websocket_routes(app: FastAPI) -> None:
    @app.websocket("/ws/generate")
    async def ws_generate(websocket: WebSocket) -> None:
        await websocket.accept()
        state = getattr(websocket.app.state, "server", None)
        # Support compat patching via pocket_tts.main globals for tests/tools.
        try:
            import pocket_tts.main as main

            model = getattr(main, "tts_model", None) or (state.model if state else None)
            default_voice_state = getattr(main, "default_voice_state", None) or (
                state.default_voice_state if state else None
            )
        except Exception:
            model = state.model if state else None
            default_voice_state = state.default_voice_state if state else None

        if model is None:
            await websocket.close(code=1013, reason="model not loaded")
            return

        stop = threading.Event()
        worker: threading.Thread | None = None
        send_lock = asyncio.Lock()
        loop = asyncio.get_running_loop()

        async def send_bytes(data: bytes) -> None:
            async with send_lock:
                await websocket.send_bytes(data)

        async def send_json(obj: dict[str, Any]) -> None:
            async with send_lock:
                await websocket.send_text(json.dumps(obj, ensure_ascii=False))

        def resolve_voice_state(voice: str | None):
            if voice is None:
                if default_voice_state is None:
                    raise ValueError("The server has no default voice loaded.")
                return default_voice_state
            if not (
                voice.startswith("http://")
                or voice.startswith("https://")
                or voice.startswith("hf://")
                or voice in _ORIGINS_OF_PREDEFINED_VOICES
            ):
                raise ValueError("voice must start with http://, https://, hf://, or be a predefined voice name")
            return model._cached_get_state_for_audio_prompt(voice)

        def generation_worker(text: str, voice: str | None) -> None:
            try:
                model_state = resolve_voice_state(voice)
                for chunk in model.generate_audio_stream(
                    model_state=model_state, text_to_generate=text, stop=stop
                ):
                    if stop.is_set():
                        break
                    pcm = _tensor_to_pcm16_bytes(chunk)
                    fut = asyncio.run_coroutine_threadsafe(send_bytes(pcm), loop)
                    fut.result()
                if not stop.is_set():
                    asyncio.run_coroutine_threadsafe(send_json({"done": True}), loop).result()
            except Exception as exc:
                if not stop.is_set():
                    try:
                        asyncio.run_coroutine_threadsafe(
                            send_json({"error": str(exc), "done": True}), loop
                        ).result(timeout=5)
                    except Exception:
                        pass
            finally:
                stop.set()

        try:
            while True:
                msg = await websocket.receive()
                if "text" in msg and msg["text"] is not None:
                    try:
                        payload = json.loads(msg["text"])
                    except json.JSONDecodeError:
                        await send_json({"error": "invalid JSON"})
                        continue
                    if not isinstance(payload, dict):
                        await send_json({"error": "JSON object required"})
                        continue

                    # Client signals cancellation/barge-in.
                    if payload.get("stop") is True or payload.get("cancel") is True:
                        stop.set()
                        if worker and worker.is_alive():
                            worker.join(timeout=5)
                        stop.clear()
                        await send_json({"stopped": True})
                        continue

                    text = payload.get("text")
                    voice = payload.get("voice")
                    flush = payload.get("flush", True)
                    if voice is not None and not isinstance(voice, str):
                        await send_json({"error": "voice must be a string"})
                        continue
                    if text is None:
                        if flush and worker and worker.is_alive():
                            # Nothing to queue in this minimal implementation.
                            pass
                        continue
                    if not isinstance(text, str) or not text.strip():
                        await send_json({"error": "text must be a non-empty string"})
                        continue
                    if worker and worker.is_alive():
                        await send_json({"error": "generation already in progress; send stop/cancel first"})
                        continue

                    stop.clear()
                    worker = threading.Thread(
                        target=generation_worker, args=(text, voice), daemon=True
                    )
                    worker.start()
                elif "bytes" in msg and msg["bytes"] is not None:
                    # Reserved for future binary control/streaming; ignore.
                    continue
                elif msg.get("type") == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            stop.set()
            if worker and worker.is_alive():
                worker.join(timeout=5)
            try:
                await websocket.close()
            except Exception:
                pass
