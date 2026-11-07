#!/usr/bin/env python3
"""Benchmark generation speed (RTF), memory, and TTFT across Pocket TTS model configs.

Runs in-process using the Python API for accurate apples-to-apples comparison.
Outputs a Markdown table.

Usage:
  uv run scripts/benchmark_models.py
  uv run scripts/benchmark_models.py --text "Custom benchmark text." --voice alba
  uv run scripts/benchmark_models.py --output benchmark_results.md
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

SAMPLE_RATE = 24000


@dataclass
class BenchResult:
    name: str
    language: str
    quantize: bool
    load_s: float | None
    rss_mb_before: float | None
    rss_mb_after_load: float | None
    rss_mb_peak_after_gen: float | None
    ttft_ms: float | None
    audio_s: float | None
    wall_s: float | None
    rtf_audio_over_wall: float | None
    chunks: int | None
    error: str | None = None


def rss_mb() -> float | None:
    try:
        import psutil  # type: ignore

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    # Windows fallback
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return counters.WorkingSetSize / (1024 * 1024)
        except Exception:
            return None
    return None


def bench_one(
    name: str,
    language: str,
    quantize: bool,
    text: str,
    voice: str,
) -> BenchResult:
    from pocket_tts import TTSModel

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rss_before = rss_mb()
    try:
        t0 = time.perf_counter()
        model = TTSModel.load_model(language=language, quantize=quantize)
        load_s = time.perf_counter() - t0
        rss_after_load = rss_mb()

        voice_state = model.get_state_for_audio_prompt(voice)
        rss_peak = rss_mb()
        t_send = time.perf_counter()
        t_first = None
        total_samples = 0
        chunks = 0
        for chunk in model.generate_audio_stream(voice_state, text):
            if t_first is None:
                t_first = time.perf_counter()
            n = chunk.shape[-1] if hasattr(chunk, "shape") else len(chunk)
            total_samples += int(n)
            chunks += 1
            r = rss_mb()
            if r is not None and (rss_peak is None or r > rss_peak):
                rss_peak = r
        t_end = time.perf_counter()
        wall_s = t_end - t_send
        audio_s = total_samples / SAMPLE_RATE
        ttft_ms = ((t_first - t_send) * 1000.0) if t_first is not None else None
        rtf = (audio_s / wall_s) if wall_s > 0 else None

        # release
        del voice_state
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return BenchResult(
            name=name,
            language=language,
            quantize=quantize,
            load_s=load_s,
            rss_mb_before=rss_before,
            rss_mb_after_load=rss_after_load,
            rss_mb_peak_after_gen=rss_peak,
            ttft_ms=ttft_ms,
            audio_s=audio_s,
            wall_s=wall_s,
            rtf_audio_over_wall=rtf,
            chunks=chunks,
        )
    except Exception as e:
        return BenchResult(
            name=name,
            language=language,
            quantize=quantize,
            load_s=None,
            rss_mb_before=rss_before,
            rss_mb_after_load=rss_mb(),
            rss_mb_peak_after_gen=rss_mb(),
            ttft_ms=None,
            audio_s=None,
            wall_s=None,
            rtf_audio_over_wall=None,
            chunks=None,
            error=str(e),
        )


def fmt(v: Any, digits: int = 2) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def to_markdown(results: list[BenchResult], text: str, voice: str) -> str:
    lines = []
    lines.append("# Pocket TTS Benchmark")
    lines.append("")
    lines.append(f"- Text: `{text}`")
    lines.append(f"- Voice: `{voice}`")
    lines.append(f"- Platform: `{sys.platform}` Python `{sys.version.split()[0]}`")
    lines.append("")
    headers = [
        "Model",
        "Language",
        "Quant",
        "Load (s)",
        "RSS after load (MB)",
        "Peak RSS (MB)",
        "TTFT (ms)",
        "Audio (s)",
        "Wall (s)",
        "RTF (audio/wall)",
        "Chunks",
        "Status",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in results:
        status = "ok" if not r.error else f"error: {r.error}"
        row = [
            r.name,
            r.language,
            "int8" if r.quantize else "fp32",
            fmt(r.load_s),
            fmt(r.rss_mb_after_load, 1),
            fmt(r.rss_mb_peak_after_gen, 1),
            fmt(r.ttft_ms, 1),
            fmt(r.audio_s),
            fmt(r.wall_s),
            fmt(r.rtf_audio_over_wall),
            fmt(r.chunks, 0),
            status.replace("|", "/"),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Notes: RTF > 1 means faster than real-time. RSS measured via `psutil` when available, else Windows process counters. TTFT is first decoded PCM chunk time in `generate_audio_stream`.")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark Pocket TTS models/configs")
    p.add_argument(
        "--text",
        default="Hello, this is a benchmark of Pocket TTS streaming latency and throughput.",
    )
    p.add_argument("--voice", default="alba")
    p.add_argument("--output", default=None, help="Write Markdown table to file")
    p.add_argument("--include-distilled", action="store_true", default=True)
    p.add_argument("--no-distilled", dest="include_distilled", action="store_false")
    args = p.parse_args()

    configs = [
        ("english_fp32", "english", False),
        ("english_int8", "english", True),
    ]
    if args.include_distilled:
        configs.extend(
            [
                ("english_18m_fp32", "english_18m_2026-10", False),
                ("english_7m_fp32", "english_7m_2026-10", False),
            ]
        )

    results = []
    for name, language, quantize in configs:
        print(f"\n=== Benchmarking {name} (language={language}, quantize={quantize}) ===", flush=True)
        r = bench_one(name, language, quantize, args.text, args.voice)
        results.append(r)
        if r.error:
            print(f"ERROR: {r.error}", flush=True)
        else:
            print(
                f"load={fmt(r.load_s)}s ttft={fmt(r.ttft_ms,1)}ms audio={fmt(r.audio_s)}s wall={fmt(r.wall_s)}s RTF={fmt(r.rtf_audio_over_wall)}x",
                flush=True,
            )

    md = to_markdown(results, args.text, args.voice)
    print("\n" + md)
    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
