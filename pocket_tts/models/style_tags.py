"""Emotion/style tag parsing and conditioning overrides for Voxenium Engine v2."""
from __future__ import annotations

import re
from dataclasses import dataclass

_TAG_RE = re.compile(r"\[(whisper|cheerful|happy|calm|serious)\]", re.IGNORECASE)


@dataclass
class StyleDirective:
    tag: str | None
    text: str
    temp_delta: float = 0.0
    noise_clamp_delta: float = 0.0
    speed_factor: float = 1.0
    frames_after_eos_bias: int = 0


_DEFAULTS = {
    "whisper": StyleDirective(tag="whisper", text="", temp_delta=-0.05, noise_clamp_delta=0.5, speed_factor=0.92, frames_after_eos_bias=1),
    "cheerful": StyleDirective(tag="cheerful", text="", temp_delta=0.04, noise_clamp_delta=-0.05, speed_factor=1.06, frames_after_eos_bias=0),
    "happy": StyleDirective(tag="happy", text="", temp_delta=0.05, noise_clamp_delta=-0.05, speed_factor=1.05, frames_after_eos_bias=0),
    "calm": StyleDirective(tag="calm", text="", temp_delta=-0.03, noise_clamp_delta=0.1, speed_factor=0.94, frames_after_eos_bias=0),
    "serious": StyleDirective(tag="serious", text="", temp_delta=-0.02, noise_clamp_delta=0.0, speed_factor=0.98, frames_after_eos_bias=0),
}


def parse_style_tags(text: str) -> StyleDirective:
    if not text:
        return StyleDirective(tag=None, text=text)
    tags = [m.group(1).lower() for m in _TAG_RE.finditer(text)]
    clean = _TAG_RE.sub(" ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    if not tags:
        return StyleDirective(tag=None, text=clean or text.strip())
    # Use first tag as primary style for predictable behavior.
    primary = tags[0]
    base = _DEFAULTS.get(primary, StyleDirective(tag=primary, text=clean))
    return StyleDirective(
        tag=base.tag,
        text=clean,
        temp_delta=base.temp_delta,
        noise_clamp_delta=base.noise_clamp_delta,
        speed_factor=base.speed_factor,
        frames_after_eos_bias=base.frames_after_eos_bias,
    )


def apply_style_to_params(
    temp: float | None,
    noise_clamp: float | None,
    speed: float,
    style: StyleDirective,
) -> tuple[float | None, float | None, float]:
    t = temp
    n = noise_clamp
    if t is not None:
        t = max(0.0, min(2.0, t + style.temp_delta))
    if n is not None:
        n = max(0.0, n + style.noise_clamp_delta)
    elif style.noise_clamp_delta > 0:
        n = style.noise_clamp_delta
    s = speed * style.speed_factor
    return t, n, s


__all__ = ["StyleDirective", "parse_style_tags", "apply_style_to_params"]
