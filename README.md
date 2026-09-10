# LectureLens 🎓👁️

> **The Vision-LLM Powered STEM Audiobook Generator.**  
> *Turn complex, math-heavy textbooks and scientific papers into spoken, professor-guided audio lectures.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Vision_LLM-Ollama_Local-black.svg)](https://ollama.com)
[![TTS: Kokoro](https://img.shields.io/badge/TTS-Kokoro--82M-green.svg)](https://github.com/hexgrad/kokoro)

---

## 👁️ Why Vision LLMs Change Everything

Traditional PDF-to-audio tools use **raw text extraction (OCR/PDF scrapers)**. On academic textbooks and STEM papers, this fails completely:
- ❌ **Broken Reading Order**: Multi-column papers get read horizontally across columns, scrambling sentences into gibberish.
- ❌ **Corrupted Equations**: Fractions like $\frac{a}{b}$ become `"a"` followed three lines later by `"b"`. Superscripts, integrals, and Greek subscripts become unpronounceable text soup.
- ❌ **Blind to Figures**: Graphs, circuit diagrams, and multi-panel figures are either skipped entirely or read as nonsense axis numbers (`0 1 2 3 -1 -2`).

### 🚀 The LectureLens Vision-First Solution:

```
┌─────────────────┐       ┌────────────────────────┐       ┌───────────────────────┐       ┌──────────────────┐
│  Textbook PDF   │  ──>  │  High-DPI Visual Page  │  ──>  │   Local Vision LLM    │  ──>  │    Kokoro TTS    │  ──>  page_0010.mp3
│ (Equations/Fig) │       │   Render (150+ DPI)    │       │ (Qwen3.8 / Qwen3.5-VL)│       │ (Neural Spoken)  │       (Playlist Ready)
└─────────────────┘       └────────────────────────┘       └───────────────────────┘       └──────────────────┘
                                                                       │
                                                       • Sees two-column flow naturally
                                                       • Translates math to spoken English
                                                       • Walks through diagrams (a, b, c, d)
                                                       • Anchors to Equation & Section #s
```

Instead of scraping fragile raw text strings, **LectureLens renders each page into a high-resolution visual snapshot and feeds it directly into a local Vision LLM** (`qwen3.8:latest`, `qwen3.5:latest`, `qwen2.5-vl`). 

The Vision model "reads" the page with **human-like spatial intelligence**:
1. **Understands Layout Hierarchy**: Reads two-column layouts top-to-bottom, left-to-right without column bleeding.
2. **Pedagogical Diagram Walkthroughs**: Identifies multi-panel figures (e.g. Figure 2.10 parts a, b, c, d) and describes the horizontal/vertical axes, waveforms, and physical intuition.
3. **Translates Formulas to Spoken English**: Turns complex notation ($x[n] \to$ *"x of n"*, $\omega_0 \to$ *"omega naught"*, $\sum \to$ *"sum from minus infinity to n"*) into fluid lecture speech.
4. **100% Local & Private**: Runs entirely on your local machine via [Ollama](https://ollama.com) with zero cloud API keys or subscriptions.

---

## 🌟 Key Features

### 1. 👁️ Multimodal Vision Reading (`--vision`)
- Local Vision inference powered by Ollama (`qwen3.8`, `qwen3.5`, `qwen2.5-vl`, etc.).
- Direct image understanding prevents formula breakage, missing symbols, and OCR distortion.

### 2. 📖 Real-Time Per-Page Audio Streaming (`--per-page`)
- Converts textbooks into discrete, numbered audio tracks (`page_0001.mp3`, `page_0002.mp3`, etc.).
- **Start listening in 15 seconds**: You don't have to wait hours for an entire book to finish before listening.
- **Auto-generated Playlist (`playlist.m3u`)**: Instantly open the output directory in VLC, Audacious, or your phone for sequential listening.
- **Crash-Resilient Resuming**: Skips already generated pages automatically if you pause or resume.

### 3. 🎯 Anchored Guided Reading (Eyes + Ears Synchronized)
- The AI professor explicitly announces section headings (*"Section 2.2: Linear Systems"*) and equation numbers (*"Looking at Equation 2.21..."*).
- Allows you to follow along visually in your printed or digital textbook without losing your place.

### 4. 🔢 Printed Page to PDF Offset Mapping (`--offset`)
- Eliminates page numbering confusion caused by Roman numeral prefaces and table of contents.
- Enter the exact printed page numbers from the book:
  ```bash
  python lecturelens.py textbook.pdf --per-page --pages 10-61 --offset 23
  ```

### 5. 📑 Multi-Range Chapter Batch Selection
- Process complex non-contiguous chapters in a single run:
  ```bash
  --pages "10-61,99-135,153-166,206-224,624-683,718-743"
  ```

### 6. 🗣️ Spoken Mathematics Engine
- Formats equations for natural spoken delivery without screen-reader syntax artifacts (*"open brace"*, *"close bracket"*, *"quote unquote"*, LaTeX dollar signs).

### 7. 🔗 Page-to-Page Continuity Memory
- Passes a trailing context window (~100 words) from Page $N-1$ into Page $N$ to complete sentences that span across page breaks.

---

## 🚀 Setup & Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/LectureLens.git
cd LectureLens
```

### Step 2: Set Up Python Environment
```bash
# Recommended: create a clean conda or venv environment
conda create -n lecturelens python=3.10 -y
conda activate lecturelens

# Install Python dependencies
pip install -r requirements.txt
```

### Step 3: Install `ffmpeg`
```bash
# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Conda
conda install -c conda-forge ffmpeg
```

### Step 4: Install Ollama & Pull a Vision LLM
1. Install [Ollama](https://ollama.com/download).
2. Pull your preferred Vision model:
```bash
ollama pull qwen3.8:latest
# or
ollama pull qwen3.5:latest
```

---

## 🎧 Usage Examples

### 📚 1. Textbook Audiobook Mode (Page-by-Page with Vision LLM)
Generate page-by-page audio for specific chapters with page offset:
```bash
python lecturelens.py textbook.pdf \
    --per-page \
    --pages "10-61,99-135" \
    --offset 23 \
    --model "qwen3.8:latest" \
    --out-dir ./Audiobook/MathChapters
```

### 📄 2. Direct Research Paper Mode (Single MP3)
Convert an entire research paper directly from an arXiv URL or local file:
```bash
# Direct arXiv URL
python lecturelens.py https://arxiv.org/pdf/2506.10947.pdf --mp3 --speed 1.2 --no-references

# Local PDF file
python lecturelens.py paper.pdf --mp3 --speed 1.2
```

### ⚡ 3. Background Batch Processing (`run.sh`)
For long overnight generation runs (e.g. 200+ pages), use `tmux`:
```bash
tmux new -s audiobook
./run.sh
# Detach anytime with Ctrl+B then D
```

---

## ⚙️ Command-Line Options

| Flag | Default | Description |
|---|---|---|
| `--per-page` | `off` | Real-time per-page audiobook mode (`page_0001.mp3`, `page_0002.mp3`, etc.) |
| `--pages RANGE` | `all` | Page selection (1-indexed), e.g. `10-61` or `"10-61,99-135"` |
| `--offset N` | `0` | Offset between printed page numbers and PDF indices (e.g. `--offset 23`) |
| `--out-dir DIR` | `./<book>/` | Output directory for `--per-page` mode |
| `--no-resume` | `off` | Overwrite existing page files instead of skipping completed pages |
| `--model MODEL`, `-m` | auto | Ollama Vision model to use (e.g. `qwen3.8:latest`, `qwen3.5:latest`) |
| `--dpi DPI` | `150` | DPI rendering resolution for Vision processing |
| `--speed X` | `1.0` | Speech speed multiplier (`0.5`–`3.0`, recommended: `1.0`–`1.3`) |
| `--voice ID` | `af_heart` | Kokoro TTS voice (run `--list-voices` to see all) |
| `--no-references` | `off` | Stop before the bibliography / references section |
| `--mp3` | `off` | Save output as MP3 (default in per-page mode) |
| `--m4a` | `off` | Save output as M4A (Apple-native format) |
| `--audio-only` | `off` | Save output as uncompressed WAV |
| `--list-voices` | — | Print all available voices and exit |

---

## 🎙️ Available Kokoro Voices

| Voice ID | Style / Persona | Accent |
|---|---|---|
| `af_heart` | Warm, natural *(default)* | American Female |
| `af_bella` | Expressive, engaging | American Female |
| `af_nicole` | Calm, soothing | American Female |
| `af_sarah` | Crisp, clear | American Female |
| `af_sky` | Bright, conversational | American Female |
| `am_adam` | Deep, resonant | American Male |
| `am_michael` | Natural, professor-like | American Male |
| `bf_emma` | Crisp, academic | British Female |
| `bf_isabella` | Soft, clear | British Female |
| `bm_george` | Authoritative, formal | British Male |
| `bm_lewis` | Casual, modern | British Male |

---

## 📜 Acknowledgements & License

- **Original Work**: Built from the work of [pdf2speech](https://github.com/soumyasj/pdf2speech) by [soumyasj](https://github.com/soumyasj).
- **Neural TTS Engine**: [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad.
- **Vision & Multimodal LLM**: Powered locally by [Ollama](https://ollama.com) and the [Qwen](https://github.com/QwenLM) model family.
- **PDF Rendering**: [PyMuPDF](https://github.com/pymupdf/PyMuPDF).

Licensed under the [MIT License](LICENSE).
