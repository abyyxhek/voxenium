#!/usr/bin/env python3
"""Real-time WebSocket streaming client for pocket-tts.

Usage examples:
  uv run --with websockets scripts/stream_client.py --text "Hello from websocket streaming."
  uv run --with websockets scripts/stream_client.py --text "Hello" --voice alba --play
  uv run --with websockets scripts/stream_client.py --simulate-llm-tokens "Hello there, this arrives like tokens."

Connects to ws://127.0.0.1:8000/ws/generate, sends {"text","voice","flush":true},
receives binary 16-bit PCM (24kHz mono) chunks, prints TTFT/RTF/jitter metrics,
optionally plays audio live (sounddevice), and writes stream_output.wav.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import queue
import statistics
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any

SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2  # int16


def pcm_duration_seconds(pcm_bytes: bytes) -> float:
    return len(pcm_bytes) / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH)


def write_wav(path: Path, pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


class OptionalPlayer:
    """Non-blocking playback via sounddevice if available."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._q: queue.Queue[bytes | None] | None = None
        self._thread: threading.Thread | None = None
        self._sd = None
        self._stream = None
        self._buf = bytearray()
        self._min_buffer_bytes = SAMPLE_RATE * SAMPLE_WIDTH // 5  # ~200ms jitter buffer

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            import sounddevice as sd  # type: ignore

            self._sd = sd
        except Exception as e:  # pragma: no cover
            print(f"[playback disabled] sounddevice unavailable: {e}", file=sys.stderr)
            self.enabled = False
            return
        self._q = queue.Queue(maxsize=200)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._q is not None and self._sd is not None
        import numpy as np  # local import; numpy is a project dependency

        def callback(outdata, frames, time_info, status):  # noqa: ARG001
            need = frames * SAMPLE_WIDTH
            while len(self._buf) < need:
                try:
                    item = self._q.get(timeout=0.05)
                except queue.Empty:
                    item = b"" if self._buf else None
                if item is None:
                    out = np.frombuffer(bytes(self._buf), dtype=np.int16).astype(np.float32) / 32768.0
                    self._buf.clear()
                    if out.size < frames:
                        pad = np.zeros(frames - out.size, dtype=np.float32)
                        out = np.concatenate([out, pad])
                    outdata[:] = out.reshape(-1, 1)
                    raise self._sd.CallbackStop()
                if item:
                    self._buf.extend(item)
                if not item and len(self._buf) >= need:
                    break
                if not item and self._q.empty():
                    break
            chunk = bytes(self._buf[:need])
            del self._buf[:need]
            if len(chunk) < need:
                chunk += b"\x00" * (need - len(chunk))
            out = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            outdata[:] = out.reshape(-1, 1)

        # Wait for initial jitter buffer.
        started = False
        primed = bytearray()
        while not started:
            item = self._q.get()
            if item is None:
                return
            primed.extend(item)
            if len(primed) >= self._min_buffer_bytes:
                self._buf.extend(primed)
                started = True
        self._stream = self._sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback, blocksize=1024
        )
        self._stream.start()
        while self._stream.active:
            time.sleep(0.05)

    def push(self, pcm: bytes) -> None:
        if self.enabled and self._q is not None:
            try:
                self._q.put(pcm, timeout=5)
            except queue.Full:
                pass

    def close(self) -> None:
        if self.enabled and self._q is not None:
            try:
                self._q.put(None, timeout=2)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass


async def receive_audio(
    uri: str,
    text: str,
    voice: str | None,
    output_path: Path,
    play: bool,
    simulate_llm_tokens: bool,
    token_delay_ms: float,
) -> dict[str, Any]:
    try:
        import websockets  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: websockets. Run with: uv run --with websockets scripts/stream_client.py ..."
        ) from e

    pcm_all = bytearray()
    chunk_arrivals: list[float] = []
    chunk_sizes: list[int] = []
    done = False
    error = None
    t_send = None
    t_first = None

    player = OptionalPlayer(play)
    player.start()

    async with websockets.connect(uri, max_size=None) as ws:
        if simulate_llm_tokens:
            # Simulate LLM token emission, but current server accepts one generation
            # request at a time and generates from full text. We buffer tokens and send
            # the final prompt when flushed (end of simulated stream).
            tokens = text.split(" ")
            buf = []
            for i, tok in enumerate(tokens):
                buf.append(tok)
                await asyncio.sleep(token_delay_ms / 1000.0)
                # Optional future protocol hint; ignored by current server.
                await ws.send(json.dumps({"text_delta": tok + (" " if i + 1 < len(tokens) else ""), "flush": False}))
            text_to_send = " ".join(buf).strip()
            payload = {"text": text_to_send, "voice": voice, "flush": True}
            t_send = time.perf_counter()
            await ws.send(json.dumps(payload))
        else:
            payload = {"text": text, "voice": voice, "flush": True}
            t_send = time.perf_counter()
            await ws.send(json.dumps(payload))

        while True:
            msg = await ws.recv()
            now = time.perf_counter()
            if isinstance(msg, (bytes, bytearray)):
                if t_first is None:
                    t_first = now
                pcm_all.extend(msg)
                chunk_arrivals.append(now)
                chunk_sizes.append(len(msg))
                player.push(bytes(msg))
            else:
                try:
                    obj = json.loads(msg)
                except Exception:
                    obj = {"raw": msg}
                if isinstance(obj, dict):
                    if obj.get("error"):
                        error = obj["error"]
                    if obj.get("done"):
                        done = True
                        break
                    if obj.get("stopped"):
                        done = True
                        break
                    # ignore text_delta ack/errors from current server
                if done:
                    break

    player.close()
    write_wav(output_path, bytes(pcm_all), SAMPLE_RATE)

    t_end = time.perf_counter()
    assert t_send is not None
    ttft_ms = ((t_first - t_send) * 1000.0) if t_first is not None else None
    total_ms = (t_end - t_send) * 1000.0
    audio_s = pcm_duration_seconds(bytes(pcm_all))
    audio_ms = audio_s * 1000.0
    rtf = (audio_s / (t_end - t_send)) if (t_end > t_send and audio_s > 0) else None

    inter_arrival = []
    if len(chunk_arrivals) >= 2:
        inter_arrival = [b - a for a, b in zip(chunk_arrivals, chunk_arrivals[1:])]
    chunk_durs = [pcm_duration_seconds(bytes(s)) * 1000.0 for s in chunk_sizes]
    metrics = {
        "uri": uri,
        "voice": voice,
        "output_path": str(output_path),
        "bytes_received": len(pcm_all),
        "chunks": len(chunk_sizes),
        "ttft_ms": ttft_ms,
        "total_time_ms": total_ms,
        "audio_duration_ms": audio_ms,
        "rtf_audio_over_wall": rtf,
        "rtf_wall_over_audio": (1.0 / rtf) if rtf else None,
        "avg_chunk_ms": statistics.mean(chunk_durs) if chunk_durs else None,
        "avg_inter_arrival_ms": statistics.mean(inter_arrival) * 1000.0 if inter_arrival else None,
        "inter_arrival_jitter_ms_std": statistics.pstdev(inter_arrival) * 1000.0 if len(inter_arrival) >= 2 else None,
        "done": done,
        "error": error,
    }
    return metrics


def print_metrics(m: dict[str, Any]) -> None:
    print("\n--- Pocket TTS Stream Metrics ---")
    print(f"URI:                {m['uri']}")
    print(f"Voice:              {m['voice']}")
    print(f"Output:             {m['output_path']}")
    print(f"Chunks:             {m['chunks']} ({m['bytes_received']} bytes PCM16)")
    if m["ttft_ms"] is not None:
        print(f"TTFT:               {m['ttft_ms']:.2f} ms")
    else:
        print("TTFT:               n/a (no audio received)")
    print(f"Total time:         {m['total_time_ms']:.2f} ms")
    print(f"Audio duration:     {m['audio_duration_ms']:.2f} ms")
    if m["rtf_audio_over_wall"]:
        print(f"RTF (audio/wall):   {m['rtf_audio_over_wall']:.3f}x")
        print(f"Inverse RTF:        {m['rtf_wall_over_audio']:.3f}x wall/audio")
    if m["avg_chunk_ms"] is not None:
        print(f"Avg chunk duration: {m['avg_chunk_ms']:.2f} ms")
    if m["avg_inter_arrival_ms"] is not None:
        print(f"Avg inter-arrival:  {m['avg_inter_arrival_ms']:.2f} ms")
    if m["inter_arrival_jitter_ms_std"] is not None:
        print(f"Arrival jitter std: {m['inter_arrival_jitter_ms_std']:.2f} ms")
    if m["error"]:
        print(f"Server error:       {m['error']}")


def main() -> int:
    p = argparse.ArgumentParser(description="Real-time streaming client for pocket-tts /ws/generate")
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws/generate", help="WebSocket URL")
    p.add_argument("--text", default=None, help="Text to synthesize")
    p.add_argument("--voice", default=None, help="Voice name/URL (server default if omitted)")
    p.add_argument("--output", default="stream_output.wav", help="Output WAV path")
    p.add_argument("--play", action="store_true", help="Live playback via sounddevice (optional dependency)")
    p.add_argument(
        "--simulate-llm-tokens",
        nargs="?",
        const="__USE_TEXT__",
        default=None,
        help="Simulate token stream input; optional value overrides --text",
    )
    p.add_argument("--token-delay-ms", type=float, default=35.0, help="Delay between simulated tokens")
    args = p.parse_args()

    text = args.text
    if args.simulate_llm_tokens is not None:
        text = args.text if args.simulate_llm_tokens == "__USE_TEXT__" else args.simulate_llm_tokens
    if not text:
        p.error("--text is required (or pass text to --simulate-llm-tokens)")

    metrics = asyncio.run(
        receive_audio(
            uri=args.url,
            text=text,
            voice=args.voice,
            output_path=Path(args.output),
            play=args.play,
            simulate_llm_tokens=args.simulate_llm_tokens is not None,
            token_delay_ms=args.token_delay_ms,
        )
    )
    print_metrics(metrics)
    return 0 if not metrics["error"] and metrics["bytes_received"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
