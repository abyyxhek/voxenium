# 🎙️ VOXENIUM

> **The Next-Generation Ultra-Lightweight, Real-Time Text-to-Speech Engine for CPUs.**  
> *Sub-100ms Latency • Offline • Zero-Gated Weights • Real-Time WebSockets • Emotion & Speed Control*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Latency](https://img.shields.io/badge/TTFT-64ms-brightgreen.svg)]()
[![Throughput](https://img.shields.io/badge/RTF-5.27x-brightgreen.svg)]()

---

## 🚀 Why Voxenium?

**Voxenium** is an ultra-lightweight, real-time Text-to-Speech engine engineered from the ground up to solve the latency, footprint, and usability bottlenecks of existing research TTS models:

* ⚡ **Ultra-Low Latency (Sub-100ms)**: Achieves **64.3 ms Time-To-First-Audio (TTFT)** on standard consumer CPUs.
* 🏎️ **5.27x Real-Time Factor (RTF)**: Generates 5 seconds of clean 24kHz audio in just 1 second on a standard CPU thread.
* 🧠 **Intelligent Text Normalizer**: Spoken-English expansion for numbers (`$50.99` ➔ *"fifty dollars and ninety nine cents"*, `1st` ➔ *"first"*, `Dr.` ➔ *"Doctor"*).
* 🎭 **Emotion & Style Tags**: Parse bracketed emotion directives on the fly: `[cheerful]`, `[whisper]`, `[calm]`, `[urgent]`.
* 🎚️ **Dynamic Speed Slider**: Real-time pacing controller (`--speed 0.8` to `1.5`) without pitch distortion.
* 🌐 **Binary WebSocket Streaming**: Low-overhead `/ws/generate` endpoint streaming raw 16-bit PCM frames directly to conversational voice agents.
* 🔓 **Zero-Auth Open Weights**: Built to run on public, open weights out of the box with zero HuggingFace token gates or 401 errors.
* 💾 **Tiny Model Footprints**: Optimized for edge deployment with **2M**, **7M**, and **18M** parameter architectures.

---

## 📊 Benchmark Results (Tested on Standard CPU)

| Model Architecture | Parameters | TTFT (First Audio) | Real-Time Factor (RTF) | Wall Time for 5.8s Audio |
| :--- | :--- | :--- | :--- | :--- |
| **`voxenium-7m`** | **6.9M** | **64.3 ms** | **5.27×** | **1.11s** |
| **`voxenium-18m`** | **18.2M** | **95.6 ms** | **4.59×** | **1.24s** |
| **`voxenium-standard`** | 100M | 269.0 ms | 1.66× | 3.51s |

---

## 📦 Quick Installation

Install with `uv` (recommended):

```bash
git clone https://github.com/abyyxhek/voxenium.git
cd voxenium
uv sync
```

---

## ⚡ Quickstart

### 1. Command Line Interface (CLI)
Generate speech instantly with intelligent normalization and speed control:

```bash
# Basic generation
uv run voxenium generate --text "Hello world! Voxenium is speaking." --output-path output.wav

# With emotion tag and speed control (1.25x faster)
uv run voxenium generate --text "[cheerful] Good morning! How can I help you today?" --speed 1.25 --output-path cheerful.wav

# Using the ultra-fast 2M parameter model
uv run voxenium generate --language english_2m_2026-10 --text "Running on a tiny CPU footprint."
```

### 2. Python API
```python
from voxenium import VoxeniumEngine

# 1. Load the lightweight CPU engine (defaults to 2M model)
engine = VoxeniumEngine.load("english_2m_2026-10")

# 2. Generate and save audio
audio = engine.speak(
    text="[cheerful] Dr. Smith received $150.50 on the 1st of November.",
    speed=1.1,
    output_path="voxenium_output.wav"
)
```

### 3. Real-Time Streaming Server & Web Audio UI
Start the local streaming server:

```bash
uv run voxenium serve --port 8000
```

* **Web UI**: Open `http://127.0.0.1:8000` for the interactive browser player with real-time Web Audio API streaming.
* **WebSocket API**: Connect your conversational AI agent (LLM ➔ TTS) to `ws://127.0.0.1:8000/ws/generate` for low-latency 16-bit PCM streaming.

---

## 🎯 Architecture Overview

```text
Text Prompt ──► [Normalizer] ──► [FlowLM Transformer] ──► [Mimi Decoder] ──► 24kHz Audio Stream
                      ▲                   ▲
Style/Speed Tags ─────┘           (KV Cache)
```

1. **`voxenium/brain/`**: Linguistic normalization, emotion tag parsing, and prosody scaling.
2. **`voxenium/core/`**: Streaming FlowLM transformer and Mimi neural audio codec.
3. **`voxenium/server/`**: FastAPI + WebSocket binary audio streaming pipeline.
4. **`voxenium/training/`**: Voice fine-tuning scripts with few-shot adaptation support.

---

## 🎙️ Fine-Tuning Your Voice

Train Voxenium on your own voice in Google Colab (with free 16GB GPU):
1. Open [`voxenium_finetune.ipynb`](voxenium_finetune.ipynb) in [Google Colab](https://colab.research.google.com).
2. Set runtime to **T4 GPU**.
3. Follow the 5 beginner-friendly cells to record, align, and train your custom voice in ~5 minutes!

---

## 📄 License
Released under the MIT License. Built with inspiration from Kyutai's CALM and Mimi research.
