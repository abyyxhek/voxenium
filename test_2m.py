# test_2m.py
# This script tests the ultra-fast 2M parameter model on your CPU.

import scipy.io.wavfile
from pocket_tts import TTSModel

print("1. Loading the lightweight 2M model onto CPU...")
# 'english_2m_2026-10' is Kyutai's tiny ~2M parameter model.
model = TTSModel.load_model("english_2m_2026-10")

print("2. Preparing the speaker voice...")
# "alba" is one of the built-in reference voices.
voice_state = model.get_state_for_audio_prompt("alba")

print("3. Generating speech...")
# This turns our text into audio numbers (samples) at 24,000 measurements per second.
audio = model.generate_audio(
    model_state=voice_state,
    text_to_generate="Hello! I am your new local voice assistant running on your CPU."
)

print("4. Saving to audio file...")
# We use scipy to save the raw audio tensor to a standard WAV file.
scipy.io.wavfile.write("hello_2m.wav", model.sample_rate, audio.numpy())

print("Done! Audio saved to hello_2m.wav")

