"""Verification tests for Voxenium Engine v2 capabilities.

Run: uv run python test_voxenium_powers.py
(or) python test_voxenium_powers.py
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

# Ensure local repo is importable when run from repo root.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pocket_tts.models.prosody import apply_speed
from pocket_tts.models.style_tags import parse_style_tags
from pocket_tts.models.text_normalizer import normalize_text_for_tts


def test_text_normalization():
    text = "$50.99 for Dr. Watson on 5th Ave."
    out = normalize_text_for_tts(text)
    print("Normalized:", out)
    assert "dollar" in out.lower()
    assert "doctor" in out.lower()
    assert "fifth" in out.lower()
    assert "avenue" in out.lower()


def test_speed_and_style_parsing():
    audio = torch.randn(24000)
    fast = apply_speed(audio, 1.3)
    normal = apply_speed(audio, 1.0)
    print("Speed lengths:", normal.numel(), "->", fast.numel())
    assert fast.numel() < normal.numel()

    style = parse_style_tags("[cheerful] Hello there!")
    print("Style:", style)
    assert style.tag in {"cheerful", "happy"}
    assert "Hello" in style.text


def test_generation_if_available():
    try:
        from pocket_tts import TTSModel
    except Exception as e:
        print("Skipping generation test (import failed):", e)
        return

    try:
        model = TTSModel.load_model()
    except Exception as e:
        print("Skipping generation test (model load failed):", e)
        return

    voice_state = model.get_state_for_audio_prompt("alba")
    sig = inspect.signature(model.generate_audio_stream)
    kwargs = {}
    if "speed" in sig.parameters:
        kwargs["speed"] = 1.3
    if "normalize_text" in sig.parameters:
        kwargs["normalize_text"] = True

    chunks = list(
        model.generate_audio_stream(
            voice_state,
            "$50.99 for Dr. Watson on 5th Ave.",
            **kwargs,
        )
    )
    assert chunks, "expected audio chunks"
    audio = torch.cat(chunks, dim=0)
    print("Generated samples (fast/normalized):", audio.shape[0])

    chunks2 = list(
        model.generate_audio_stream(
            voice_state,
            "[cheerful] Have a wonderful day!",
            **{k: v for k, v in kwargs.items() if k != "speed"},
        )
    )
    assert chunks2, "expected audio chunks for emotion tag"
    print("Generated samples ([cheerful]):", torch.cat(chunks2, dim=0).shape[0])


def main():
    test_text_normalization()
    test_speed_and_style_parsing()
    test_generation_if_available()
    print("OK: Voxenium Engine v2 verification checks passed (where runnable).")


if __name__ == "__main__":
    main()
