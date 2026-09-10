# LectureLens 🎓📖

> **Turn technical textbooks and STEM research papers into crystal-clear, spoken professor audiobooks.**

**LectureLens** is a multimodal AI tool that transforms complex, math-heavy textbooks, multi-column scientific PDFs, and research papers into natural, engaging audio lectures. Powered by **Local Vision LLMs** (via [Ollama](https://ollama.com)) and high-speed neural TTS ([Kokoro](https://github.com/hexgrad/kokoro)), it operates 100% offline with zero cloud API keys.

---

## 🌟 Key Features

### 1. 📖 Real-Time Per-Page Streaming (`--per-page`)
- Converts textbooks into discrete, numbered audio tracks (`page_0001.mp3`, `page_0002.mp3`, etc.).
- **Start listening in 15 seconds**: You don't have to wait for an entire 800-page book to process.
- **Automatic Playlist Generation (`playlist.m3u`)**: Instantly load the whole chapter into VLC, Audacious, or your phone for sequential listening.
- **Crash-Resilient Resuming**: Skips already completed pages automatically so you can resume interrupted batch jobs anytime.

### 2. 🔢 Printed Page to PDF Offset Mapping (`--offset`)
- Solves the classic problem where printed book page numbers differ from PDF index numbers due to front matter (preface, TOC).
- Enter the exact printed page numbers from the book:
  ```bash
  python lecturelens.py textbook.pdf --per-page --pages 10-61 --offset 23
  ```

### 3. 📑 Multi-Range & Chapter Batch Selection
- Specify multiple chapters or non-contiguous page ranges in a single run:
  ```bash
  --pages "10-61,99-135,153-166,206-224,624-683,718-743"
  ```

### 4. 👁️ Local Vision LLM Engine (`--vision`)
- Uses local Vision models (`qwen3.8:latest`, `qwen3.5:latest`, `qwen2.5-vl`) to visually read pages.
- Eliminates OCR errors on complex multi-column layouts, sidebars, matrices, and equations.

### 5. 🎯 Anchored Guided Reading & Diagram Pedagogy
- **Synchronized Eyes & Ears**: The audio acts as a personal tutor, explicitly announcing section headings (*"Section 2.2: Linear Systems"*) and equation numbers (*"Looking at Equation 2.21..."*).
- **Sub-Panel Diagram Walkthroughs**: Walks through multi-part figures (parts a, b, c, d), explaining horizontal and vertical axes, signal waveforms, and physical interpretations.

### 6. 🗣️ Spoken Mathematics Translation Engine
- Translates formulas into fluent spoken lecture English:
  - $x[n]$ $\to$ *"x of n"*
  - $h[n-k]$ $\to$ *"h of n minus k"*
  - $\omega_0$ $\to$ *"omega naught"*
  - $\sum_{k=-\infty}^{n} x[k]$ $\to$ *"sum from k equals minus infinity to n of x of k"*
- Strips screen-reader artifacts (*"open brace"*, *"bracket"*, *"quote unquote"*, raw LaTeX syntax).

### 7. 🔗 Page-to-Page Continuity & Memory Window
- Passes the ending context (~100 words) from Page $N-1$ into Page $N$.
- Seamlessly completes sentences and equations that get cut in half across page turns.

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

# Install dependencies
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

### Step 4: Install Ollama & Pull a Vision Model
1. Install [Ollama](https://ollama.com/download).
2. Pull a vision-capable model:
```bash
ollama pull qwen3.8:latest
# or
ollama pull qwen3.5:latest
```

---

## 🎧 Usage Examples

### 📚 1. Textbook Audiobook Mode (Page-by-Page)
Generate audio page-by-page for specific chapters with page offset:
```bash
python lecturelens.py textbook.pdf \
    --per-page \
    --pages "10-61,99-135" \
    --offset 23 \
    --model "qwen3.8:latest" \
    --out-dir ./Audiobook/Chapters
```

### 📄 2. Direct Research Paper Mode (Single MP3)
Convert an entire academic paper directly into a single MP3:
```bash
# From arXiv link
python lecturelens.py https://arxiv.org/pdf/2506.10947.pdf --mp3 --speed 1.2 --no-references

# From local PDF
python lecturelens.py paper.pdf --mp3 --speed 1.2
```

### ⚡ 3. Overnight Batch Script (`run.sh`)
For long batch jobs (e.g., 200+ pages), run inside `tmux`:
```bash
tmux new -s audiobook
./run.sh
# Detach with Ctrl+B then D
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
| `--model MODEL`, `-m` | auto | Ollama vision model (e.g. `qwen3.8:latest`, `qwen3.5:latest`) |
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

- **Foundational Inspiration**: Built upon concepts from [pdf2speech](https://github.com/soumyasj/pdf2speech) by soumyasj.
- **Neural TTS Engine**: [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad.
- **Vision Inference**: [Ollama](https://ollama.com) and the [Qwen](https://github.com/QwenLM) model family.
- **PDF Engine**: [PyMuPDF](https://github.com/pymupdf/PyMuPDF).

Licensed under the [MIT License](LICENSE).
