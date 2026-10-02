# 🎙️ Transcribe Master — Whisper OpenVINO (Beta)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2025%2B-purple.svg)](https://github.com/openvinotoolkit/openvino)
[![Model](https://img.shields.io/badge/Model-Whisper%20Large%20v3%20Turbo%20FP16-green.svg)](https://huggingface.co/openai/whisper-large-v3-turbo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[🇮🇷 نسخه فارسی (Persian Version)](README_FA.md)

An ultra-fast, lightweight CLI utility for transcribing audio and video files into full text (`TXT`) and synchronized subtitles (`SRT`), powered by the **Intel OpenVINO** inference engine and the **Whisper Large V3 Turbo (FP16)** model.

---

## ⚡ Hardware Compatibility

Built with an automated hardware selection and fallback pipeline (`GPU -> NPU -> CPU`):

- **Optimized for Intel Hardware:** Maximum hardware acceleration on integrated and discrete **Intel Arc GPUs (Xe / Xe2 architecture, including Arc 140V on Core Ultra / Lunar Lake series)** and **NPUs (Intel AI Boost)**.
- **Universal Fallback for All Systems:** Gracefully falls back to **CPU** with zero crashes on non-Intel systems. Fully compatible with **AMD Ryzen** and modern x86/ARM processors via optimized vectorized instruction sets.

---

## ✨ Key Features & Engineering Highlights

- **Zero-RAM Leak Audio Streaming:** Unlike traditional scripts that load multi-gigabyte audio files into memory, audio is streamed directly from disk in sliding windows. Memory footprint remains minimal (**under a few megabytes**) even on multi-hour lectures.
- **Multi-Segment Boundary Deduplication:** Uses a **30-second sliding window with a 2-second overlap**, paired with a context-aware 3-segment deduplication algorithm (handling English and Persian punctuation) to eliminate repeated words across chunk boundaries.
- **Language Drift Lock:** In auto-detect mode, the detected language is locked on the first voiced chunk, preventing erratic mid-lecture language switching during pauses or technical/medical terminology.
- **Smart Silence & Hallucination Gating:** Real-time RMS/peak-energy evaluation skips silent stretches, completely suppressing phantom Whisper hallucinations (e.g. "Thank you", "Thanks for watching").
- **Broadcast-Grade Subtitle Splitting:** Automatically splits over-long subtitle blocks into readable segments (max ~7 seconds / ~120 characters) adhering to professional subtitling guidelines.
- **SRT Timestamp Precision:** Resolves the classic millisecond carry bug (e.g. invalid `00:00:60,000` timestamps).

---

## 📥 Prerequisites & Installation

### 1. Install FFmpeg (Required for audio/video demuxing)
On Windows (PowerShell):
```powershell
winget install Gyan.FFmpeg
```

### 2. Setup Virtual Environment
```powershell
# Clone the repository
git clone https://github.com/alrzfrscn/Transcribe-Master.git
cd Transcribe-Master

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Model Weights (One-time setup)
Fetch the official FP16 OpenVINO model (~1.6GB) from Hugging Face Hub:
```powershell
python download_model.py
```

---

## 🚀 Usage Guide

### 1. Interactive Mode (Default & Recommended):
Run without arguments to interactively enter the file path and select a language from the menu (Default: English):
```powershell
python run.py
```

### 2. Direct File Input (Language Menu Prompt):
```powershell
python run.py "C:\Media\lecture.mp4"
```

### 3. Headless CLI / Automation:
```powershell
# Transcribe to English (Default)
python run.py lecture.mp4 --lang en

# Transcribe to Persian
python run.py lecture.mp4 --lang fa

# Auto-detect language with custom output directory
python run.py lecture.mp4 --lang auto --output-dir ./results

# Explicit execution on CPU
python run.py lecture.mp4 --lang en --device CPU
```

---

## 📂 Output Files

By default, outputs are saved alongside the source media file:
- **`filename.txt`** — Clean, continuous transcript without boundary repetitions.
- **`filename.srt`** — Standard SRT subtitle file with sequential numbering and precise timestamps.

---

## ⚙️ CLI Parameters

| Flag | Default | Description |
| :--- | :---: | :--- |
| `input` / `--input` | `None` | Path to media file (prompts if omitted) |
| `--lang` | Interactive (Menu default: `en`) / `en` | Speech language code (`en`, `fa`, `ar`, `tr`, `auto`, or any Whisper code) |
| `--device` | `auto` | Target execution device (`auto`, `GPU`, `NPU`, `CPU`) |
| `--model-dir` | `./whisper-large-v3-turbo-fp16-ov` | Directory containing OpenVINO model files |
| `--output-dir` | Source file directory | Custom directory for output files |
| `--chunk-duration` | `30.0` | Sliding window length in seconds |
| `--overlap` | `2.0` | Window overlap duration in seconds for seamless word stitching |
| `--no-srt-split` | `False` | Disable automatic splitting of long subtitle segments |

---

## 🗂️ Repository Structure

```text
├── run.py                 # Main unified execution pipeline
├── download_model.py      # Automated model downloader
├── requirements.txt       # Python dependencies
├── .gitignore             # Excludes model weights, caches, and virtual env
├── README.md              # English documentation (Primary)
└── README_FA.md           # Persian documentation
```

---

## 📝 License
This project is licensed under the **MIT License** — free for personal and commercial use.
