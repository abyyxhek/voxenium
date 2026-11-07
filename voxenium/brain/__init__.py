"""
Voxenium Brain: Linguistic normalization, emotional conditioning, and prosody control.
"""

from pocket_tts.models.text_normalizer import normalize_text_for_tts
from pocket_tts.models.style_tags import parse_style_tags, apply_style_to_params, StyleDirective
from pocket_tts.models.prosody import apply_speed, clamp_speed

__all__ = [
    "normalize_text_for_tts",
    "parse_style_tags",
    "apply_style_to_params",
    "StyleDirective",
    "apply_speed",
    "clamp_speed",
]
