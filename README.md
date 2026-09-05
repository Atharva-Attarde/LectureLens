# Pdf2Speech

Convert scientific PDFs into audio — listen to papers like a podcast.

Pdf2Speech extracts and cleans text from a PDF (handling multi-column layouts, math, citations, and section headings), synthesises natural-sounding speech with the [Kokoro TTS](https://github.com/hexgrad/kokoro) engine, and saves the result as an **MP3**, **M4A**, **WAV**, or **MP4** with a cover image. It also features a **podcast mode** that uses an LLM to write a HOST/GUEST interview script and synthesises it with two distinct voices. It works fully offline on CPU or GPU and accepts both local files and direct URLs.

---

## Demo

```
python pdf2speech.py https://arxiv.org/pdf/2506.10947.pdf --speed 1.3 --no-references --mp3
```

```
Downloading PDF from:
  https://arxiv.org/pdf/2506.10947.pdf  100%

  PDF  : 2506.10947.pdf
  Out  : 2506.10947.mp3
  Voice: af_heart  |  Speed: 1.3x
  Skip references: True

Step 1/3  Extracting text from PDF...
  Title : Spurious Rewards: Rethinking Training Signals in RLVR
  Length: 39,628 chars  (~6,055 words)

Step 2/3  Preprocessing text...
  5,777 words  →  estimated audio length: ~30 min

Step 3/3  Synthesising speech...
  [124/124] ...
Audio: 44.6 min  →  2506.10947.mp3

Done!  →  2506.10947.mp3
```

**Podcast mode** (AI-generated HOST/GUEST interview + two-voice synthesis):

```
python pdf2speech.py https://arxiv.org/pdf/2506.10947.pdf --speed 1.3 --no-references --podcast
```

```
  PDF  : 2506.10947.pdf
  Out  : 2506.10947.mp3  +  2506.10947_podcast.mp3
  Mode : direct transcription  +  AI podcast (Q&A)

Step 3/3  Synthesising speech (direct transcription)...
Audio: 44.6 min  →  2506.10947.mp3

Step 4/4  Generating podcast version...
  Model : Qwen/Qwen2.5-7B-Instruct
  Generating podcast script...

HOST: Welcome everyone to Science Unveiled...
GUEST: Thanks for having me! RLVR stands for...
...

  20 turns  (host=af_bella, guest=am_michael)
  Podcast audio: 3.9 min
MP3: 2506.10947_podcast.mp3

  Direct transcription  →  2506.10947.mp3
  Podcast (Q&A)         →  2506.10947_podcast.mp3
```

---

## Features

- **URL support** — pass an `https://` link directly, no manual download needed
- **11 voices** — American and British English, male and female
- **Adjustable speed** — 0.5× to 3.0× (try 1.2–1.4 for comfortable listening)
- **Smart text cleaning** — removes citations, math expressions, figure references, page numbers, and email addresses; fixes hyphenated line-breaks; announces section headings
- **Real-Time Per-Page Textbook Audiobook (`--per-page`)** — Perfect for heavy textbooks and papers. Processes pages sequentially and saves each page's MP3 (`page_0001.mp3`, `page_0002.mp3`, etc.) immediately into a book folder so you can start listening to Page 1 right away without waiting for the whole book to finish. Automatically creates a `playlist.m3u` playlist and supports resume.
- **Vision Mode (`--vision`)** — Renders each PDF page as an image and uses a Vision LLM (`qwen3.5` via Ollama) to read technical textbooks with complex multi-column layouts, sidebars, matrices, and intricate mathematical equations without OCR/layout corruption.
- **LLM Natural Narration (`--llm-clean`)** — Uses your local **Ollama** (e.g., `qwen3.5:latest` / `qwen3:8b`, 4-bit quantized, minimal VRAM, no huge downloads) or HuggingFace to clean and verbalize the complete paper for human-like narration: strips chart axes/table fragments and translates formulas into natural spoken English while preserving all technical substance (no unwanted summarization)
- **Ollama Integration** — Automatically detects and uses your local Ollama server and models (`qwen3.5`, `qwen3`, etc.) with zero setup required
- **Skip references** — optionally stop before the bibliography
- **Multiple output formats** — MP3, M4A (Apple-native), WAV, or MP4 with cover image
- **Podcast mode** — LLM generates a HOST/GUEST interview script (Qwen2.5-7B-Instruct); two Kokoro voices synthesise it into a compact podcast MP3 (`<stem>_podcast.mp3`)
- **Runs locally** — no API keys, no internet required (after first model download)
- **CPU + GPU** — Kokoro runs on CPU; GPU accelerates synthesis on larger papers

---

## Installation

**Requirements:** Python 3.10+, ffmpeg

```bash
git clone https://github.com/soumyasj/pdf2speech.git
cd pdf2speech
pip install -r requirements.txt
```

**ffmpeg** (needed for MP3/M4A/MP4 output):

```bash
# conda
conda install -c conda-forge ffmpeg

# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg
```

Or run the helper script:

```bash
bash install.sh
```

---

## Usage

```
python pdf2speech.py <pdf-or-url> [options]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--per-page` | off | Real-time per-page audiobook mode: writes `page_0001.mp3`, `page_0002.mp3`, etc. so you can start listening immediately |
| `--out-dir DIR` | `./<book_name>/` | Output directory for `--per-page` mode |
| `--pages RANGE` | all | Page selection (1-indexed), e.g. `--pages 1-10` or `--pages 1,3,5` |
| `--max-pages N` | all | Process only the first N pages |
| `--no-resume` | off | Force regenerate existing page files in `--per-page` mode |
| `--vision` | off | Render pages as images and use Ollama Vision LLM (best for math/textbooks) |
| `--llm-clean` | off | Clean and verbalize text using local Ollama LLM |
| `--speed X` | `1.0` | Speech speed multiplier (0.5–3.0, try 1.2–1.4) |
| `--voice ID` | `af_heart` | Kokoro voice (see `--list-voices`) |
| `--no-references` | off | Stop before the References section |
| `--mp3` | off | Save as MP3 (recommended for Mac / phones) |
| `--m4a` | off | Save as M4A/AAC (native Apple format) |
| `--audio-only` | off | Save as WAV (lossless, large) |
| `-o PATH` | auto | Custom output path |
| `--list-voices` | — | Print available voices and exit |
| `--podcast` | off | Generate direct MP3 **and** a Q&A podcast MP3 (`<stem>_podcast.mp3`) via LLM |
| `--podcast-model MODEL` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model for script generation |
| `--host-voice ID` | `af_bella` | Kokoro voice for the podcast HOST |
| `--guest-voice ID` | `am_michael` | Kokoro voice for the podcast GUEST |

### Examples

```bash
# Local file → MP4 with cover image (default)
python pdf2speech.py paper.pdf

# arXiv URL → MP3, 30% faster, skip references
python pdf2speech.py https://arxiv.org/pdf/2506.10947.pdf --speed 1.3 --no-references --mp3

# British male voice → M4A (plays in QuickTime / iTunes)
python pdf2speech.py paper.pdf --voice bm_george --m4a

# Custom output path
python pdf2speech.py paper.pdf -o ~/Desktop/paper_audio.mp3 --mp3

# List all voices
python pdf2speech.py --list-voices

# Generate direct MP3 + AI podcast (two-voice Q&A interview)
python pdf2speech.py paper.pdf --podcast

# Podcast with custom voices and a larger LLM
python pdf2speech.py paper.pdf --podcast --host-voice bf_emma --guest-voice bm_george --podcast-model Qwen/Qwen2.5-14B-Instruct
```

---

## Available Voices

| ID | Description |
|----|-------------|
| `af_heart` | American Female — warm *(default)* |
| `af_bella` | American Female — expressive |
| `af_nicole` | American Female — calm |
| `af_sarah` | American Female — clear |
| `af_sky` | American Female — bright |
| `am_adam` | American Male — deep |
| `am_michael` | American Male — natural |
| `bf_emma` | British Female — crisp |
| `bf_isabella` | British Female — soft |
| `bm_george` | British Male — authoritative |
| `bm_lewis` | British Male — casual |

---

## Output Formats

| Flag | Format | Notes |
|------|--------|-------|
| *(default)* | `.mp4` | Video with cover image, good for YouTube / media players |
| `--mp3` | `.mp3` | Universally compatible, recommended for most use cases |
| `--m4a` | `.m4a` | Native Apple format, plays in QuickTime and iTunes |
| `--audio-only` | `.wav` | Lossless, no re-encoding, largest file size |
| `--podcast` | `<stem>.mp3` + `<stem>_podcast.mp3` | Direct MP3 and a two-voice AI podcast |

---

## How It Works

1. **Extract** — PyMuPDF reads the PDF block-by-block, preserving column order
2. **Clean** — regex pipeline removes noise (citations, math, URLs, page numbers) and announces section headings
3. **Synthesise** — text is split into sentence-aligned chunks and fed to [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) TTS
4. **Encode** — ffmpeg combines the audio into the chosen output format; MP4 gets a generated cover image
5. *(Podcast mode)* **Script** — a Qwen2.5 instruct model writes a 13–15 exchange HOST/GUEST conversation covering key ideas, methodology, and implications; a second Kokoro pass synthesises it with two distinct voices into `<stem>_podcast.mp3`

---

## Dependencies

| Package | Purpose |
|---------|---------|
| [PyMuPDF](https://pymupdf.readthedocs.io/) | PDF text extraction |
| [Kokoro](https://github.com/hexgrad/kokoro) | TTS synthesis |
| [soundfile](https://python-soundfile.readthedocs.io/) | WAV I/O |
| [Pillow](https://python-pillow.org/) | Cover image generation |
| [ffmpeg](https://ffmpeg.org/) | Audio/video encoding |
| [transformers](https://github.com/huggingface/transformers) | Qwen LLM for podcast script generation *(podcast mode only)* |
| [torch](https://pytorch.org/) | Model inference *(podcast mode only)* |

---

## License

MIT
