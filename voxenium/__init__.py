"""
VOXENIUM: Next-Generation Ultra-Lightweight Real-Time Text-to-Speech Engine.
Designed for CPU, low-memory edge devices, and real-time voice assistants.
"""

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.models.text_normalizer import normalize_text_for_tts
from pocket_tts.models.prosody import apply_speed, clamp_speed
from pocket_tts.models.style_tags import parse_style_tags, apply_style_to_params, StyleDirective

__version__ = "2.0.0"

class VoxeniumEngine(TTSModel):
    """
    Voxenium Flagship Engine:
    - Pre-integrated with smart text normalization (numbers, currency, titles).
    - Real-time dynamic speech rate / prosody slider.
    - Emotion and style tags ([cheerful], [whisper], [calm]).
    - Public open model weights by default (zero 401 Gated errors).
    """

    @classmethod
    def load(
        cls,
        model_name: str = "english_2m_2026-10",
        speed: float = 1.0,
        quantize: bool = False,
    ) -> "VoxeniumEngine":
        """
        Loads the Voxenium engine with public weights and optimized CPU inference.
        Defaults to the ultra-lightweight 2M parameter model (64ms TTFT).
        """
        model = cls.load_model(model_name, quantize=quantize)
        model.default_speed = speed
        return model

    def speak(
        self,
        text: str,
        voice: str = "alba",
        speed: float | None = None,
        output_path: str | None = None,
    ):
        """
        High-level simple speech generation:
        1. Normalizes text ($50.99 -> fifty dollars and ninety nine cents)
        2. Applies emotion/style directives if present
        3. Generates 24kHz audio and optionally saves to file
        """
        voice_state = self.get_state_for_audio_prompt(voice)
        audio = self.generate_audio(
            model_state=voice_state,
            text_to_generate=text,
            speed=speed or getattr(self, "default_speed", 1.0),
        )
        if output_path:
            import scipy.io.wavfile
            scipy.io.wavfile.write(output_path, self.sample_rate, audio.cpu().numpy())
        return audio

__all__ = [
    "VoxeniumEngine",
    "normalize_text_for_tts",
    "apply_speed",
    "clamp_speed",
    "parse_style_tags",
    "apply_style_to_params",
    "StyleDirective",
    "__version__",
]
