"""Dynamic speech rate / prosody post-processing for decoded audio chunks."""
from __future__ import annotations

import math

import torch


def clamp_speed(speed: float) -> float:
    try:
        s = float(speed)
    except Exception:
        return 1.0
    if not math.isfinite(s):
        return 1.0
    return max(0.5, min(2.0, s))


def apply_speed(audio: torch.Tensor, speed: float) -> torch.Tensor:
    """Time-stretch/pace audio by resampling.

    speed > 1.0 => shorter/faster output; speed < 1.0 => longer/slower output.
    Uses linear interpolation for low overhead. (This affects pitch slightly;
    acceptable for lightweight real-time prosody control without extra deps.)
    """
    speed = clamp_speed(speed)
    if speed == 1.0 or audio.numel() == 0:
        return audio

    x = audio
    orig_shape = x.shape
    if x.dim() == 1:
        x2 = x.unsqueeze(0).unsqueeze(0)  # [1,1,T]
        squeeze_out = lambda y: y.squeeze(0).squeeze(0)
    elif x.dim() == 2:
        x2 = x.unsqueeze(0)  # [1,C,T]
        squeeze_out = lambda y: y.squeeze(0)
    elif x.dim() == 3:
        x2 = x
        squeeze_out = lambda y: y
    else:
        # flatten leading dims, keep last as time
        t = x.shape[-1]
        x2 = x.reshape(-1, 1, t)
        squeeze_out = lambda y: y.reshape(*orig_shape[:-1], y.shape[-1])

    t_in = x2.shape[-1]
    t_out = max(1, int(round(t_in / speed)))
    if t_out == t_in:
        return audio
    y = torch.nn.functional.interpolate(x2, size=t_out, mode="linear", align_corners=False)
    return squeeze_out(y)


__all__ = ["apply_speed", "clamp_speed"]
