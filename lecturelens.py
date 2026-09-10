#!/usr/bin/env python3
"""
LectureLens — Turn Textbooks & Scientific Papers into Spoken Professor Lectures.

Uses Local Multimodal Vision LLMs (via Ollama) and Kokoro Neural TTS.
Features Anchored Guided Reading, Spoken Mathematics verbalization,
Real-Time Per-Page Audiobook streaming, and Page-to-Page Continuity Memory.
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from tqdm import tqdm


# ---------------------------------------------------------------------------
# URL Download Helper
# ---------------------------------------------------------------------------

def download_pdf(url: str, dest_path: str):
    """Download a PDF from a URL with a visual progress bar."""
    print(f"\nDownloading PDF from:\n  {url}")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LectureLens/1.0"}
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("content-length", 0))
        block_size = 65536
        with open(dest_path, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc="  Downloading", ncols=80) as pbar:
                while True:
                    chunk = resp.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
    print("  Download complete.\n")


# ---------------------------------------------------------------------------
# PDF Layout & Column Handling
# ---------------------------------------------------------------------------

def _block_font_sizes(page) -> dict:
    """Map block index -> maximum font size across all spans in that block."""
    sizes = {}
    for block in page.get_text("dict", flags=0).get("blocks", []):
        if block.get("type") == 0:
            max_s = 0.0
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("size", 0) > max_s:
                        max_s = span["size"]
            sizes[block.get("number", -1)] = max_s
    return sizes


def _detect_column_margins(pages_blocks: list, page_width: float) -> tuple[float, float] | None:
    """Detect left and right column boundaries in a two-column academic layout."""
    mid = page_width / 2.0
    left_xs, right_xs = [], []
    for blocks in pages_blocks:
        for b in blocks:
            x0, x1 = b[0], b[2]
            w = x1 - x0
            if w > page_width * 0.65:
                continue
            if x0 < mid and x1 <= mid + page_width * 0.08:
                left_xs.append(x0)
            elif x0 >= mid - page_width * 0.08:
                right_xs.append(x0)

    if len(left_xs) < 4 or len(right_xs) < 4:
        return None
    left_xs.sort()
    right_xs.sort()
    return left_xs[len(left_xs) // 10], right_xs[len(right_xs) // 10]


def _sort_blocks_two_column(blocks: list, page_width: float, margins: tuple[float, float] | None = None) -> list:
    """Sort blocks top-to-bottom, reading left column before right column."""
    if not blocks:
        return []
    mid = page_width / 2.0
    full_width, left_col, right_col = [], [], []

    for b in blocks:
        x0, x1 = b[0], b[2]
        w = x1 - x0
        if w > page_width * 0.62:
            full_width.append(b)
        elif x0 < mid and x1 <= mid + page_width * 0.08:
            left_col.append(b)
        elif x0 >= mid - page_width * 0.08:
            right_col.append(b)
        else:
            if (x0 + x1) / 2.0 < mid:
                left_col.append(b)
            else:
                right_col.append(b)

    full_width.sort(key=lambda b: b[1])
    left_col.sort(key=lambda b: b[1])
    right_col.sort(key=lambda b: b[1])

    result = []
    fw_idx = 0
    for col_block in left_col + right_col:
        while fw_idx < len(full_width) and full_width[fw_idx][1] < col_block[1]:
            result.append(full_width[fw_idx])
            fw_idx += 1
        result.append(col_block)
    result.extend(full_width[fw_idx:])
    return result


def parse_page_selection(pages_str: str | None, max_pages: int | None, total_pages: int, offset: int = 0) -> list[tuple[int, int]]:
    """
    Parse page specification (e.g. '10-61,99-135,153-166') with an optional printed book offset.
    Returns list of (book_page_number, 0_indexed_pdf_page_index).
    """
    items = []
    if pages_str:
        selected_pages = []
        for part in pages_str.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-", 1)
                selected_pages.extend(range(int(start), int(end) + 1))
            else:
                selected_pages.append(int(part))

        seen = set()
        for p in selected_pages:
            if p not in seen:
                seen.add(p)
                pdf_idx = p + offset - 1
                if 0 <= pdf_idx < total_pages:
                    items.append((p, pdf_idx))
    elif max_pages is not None and max_pages > 0:
        for p in range(1, min(max_pages, total_pages) + 1):
            pdf_idx = p + offset - 1
            if 0 <= pdf_idx < total_pages:
                items.append((p, pdf_idx))
    else:
        for p in range(1, total_pages + 1):
            pdf_idx = p + offset - 1
            if 0 <= pdf_idx < total_pages:
                items.append((p, pdf_idx))
    return items


def extract_text_from_pdf(pdf_path: str,
                          skip_references: bool = True,
                          page_indices: list[int] | None = None) -> tuple[str, str]:
    """Extract clean text from a PDF with two-column sorting."""
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    doc = fitz.open(pdf_path)
    total_doc_pages = len(doc)
    target_indices = page_indices if page_indices is not None else list(range(total_doc_pages))

    pages_blocks = []
    selected_pages = []
    for idx in target_indices:
        if 0 <= idx < total_doc_pages:
            page = doc[idx]
            selected_pages.append(page)
            raw_blocks = page.get_text("blocks", sort=False)
            font_sizes = _block_font_sizes(page)
            pages_blocks.append([
                b + (font_sizes.get(b[5], 0),)
                for b in raw_blocks if b[6] == 0 and b[4].strip()
            ])

    page_width = doc[0].rect.width if len(doc) else 0
    margins = _detect_column_margins(pages_blocks, page_width) if page_width else None

    all_blocks = []
    for page, text_blocks in zip(selected_pages, pages_blocks):
        if not text_blocks:
            continue
        for b in _sort_blocks_two_column(text_blocks, page.rect.width, margins):
            txt = b[4].strip()
            if txt:
                all_blocks.append(txt)

    doc.close()

    title = all_blocks[0][:120].strip() if all_blocks else "Technical Paper"
    raw = "\n\n".join(all_blocks)

    if skip_references:
        patterns = [
            r'\n\s*References\s*\n',
            r'\n\s*REFERENCES\s*\n',
            r'\n\s*Bibliography\s*\n',
            r'\n\s*BIBLIOGRAPHY\s*\n',
        ]
        for pat in patterns:
            m = re.search(pat, raw)
            if m:
                raw = raw[:m.start()]
                break

    return title, raw


# ---------------------------------------------------------------------------
# Vision LLM & Spoken Math Prompting
# ---------------------------------------------------------------------------

_VISION_PROMPT = """\
You are an expert STEM professor delivering a crystal-clear, engaging audio lecture to a student who is reading this textbook page with their eyes while listening to your voice.

Your goal is Anchored Guided Learning—keeping the student's eyes synchronized with the page while thoroughly explaining concepts, equations, and diagrams:
1. PAGE & FIGURE ANCHORING:
   - Follow the physical top-to-bottom layout, section headings, example titles, and paragraphs in exact order.
   - Reference section titles (e.g. "Section 2.2.2: Linear Systems") and equation numbers (e.g. "Looking at Equation 2.21...") so the student's eyes lock onto the right block.
   - When encountering a diagram, figure, or plot (e.g. "Figure 2.10"):
     * Announce the figure title and caption clearly.
     * Walk through each sub-panel (part a, part b, part c, part d) explaining the horizontal and vertical axes, signals plotted, and what physical or mathematical behavior the visual illustrates.
2. INTUITIVE SPOKEN MATHEMATICS:
   - Speak equations with fluid, natural lecture cadence:
     * Standard signals: "x of n", "y of n", "x of k", "omega naught".
     * Operations: "x of n minus k", "sum from k equals minus infinity to infinity", "a to the power k".
     * State formulas and their physical meaning smoothly without robotic LaTeX symbols.
   - NEVER say structural coding/syntax tokens aloud (do NOT say "open parenthesis", "close parenthesis", "bracket", "open brace", "close brace", "quote", or "unquote").
   - Write out ALL Greek letters and math symbols in plain English words (no LaTeX $, $$, or backslashes).
3. PURE SPEECH:
   - Output ONLY pure, speakable English text without markdown formatting, thinking tags, or conversational introductory greetings."""


def build_vision_prompt(prev_context: str | None = None) -> str:
    """Build the vision instruction prompt with optional previous page continuity context."""
    base_prompt = _VISION_PROMPT
    if not prev_context or not prev_context.strip():
        return base_prompt

    words = prev_context.strip().split()
    snippet = " ".join(words[-100:]) if len(words) > 100 else " ".join(words)

    context_block = f"""\

[CONTINUITY CONTEXT FROM PREVIOUS PAGE]:
"...{snippet}..."

CONTINUITY INSTRUCTIONS:
- If this page begins with an incomplete sentence or ongoing derivation from the previous page, complete the thought seamlessly.
- Maintain consistent notation and terminology from the previous page, then proceed directly to guide the student through the current page."""

    return f"{base_prompt}\n\n{context_block}"


def clean_spoken_math_text(text: str) -> str:
    """Post-process LLM-generated spoken math to eliminate accidental screen-reader artifacts."""
    # Strip markdown fences and asterisks
    text = re.sub(r'^```[\w]*\n?', '', text)
    text = re.sub(r'\n?```$', '', text)
    text = re.sub(r'[*_]{1,3}', '', text)

    # Replace 'T open brace' or 'T {' with 'T of '
    text = re.sub(r'(?i)\bT\s*(?:open\s+brace|\{)\b', 'T of ', text)

    # Strip literal syntax phrases like 'open brace', 'close parenthesis', 'quote', 'unquote'
    text = re.sub(r'(?i)\b(?:open|close|left|right)\s*(?:brace|bracket|parenthes[ie]s|paren)s?\b', '', text)
    text = re.sub(r'(?i)\b(?:quote|unquote)\b', '', text)

    # Clean redundant 'times' between single variable coefficients like 'a times x' -> 'a x'
    text = re.sub(r'(?i)\b([a-zA-Z0-9])\s+times\s+([a-zA-Z])\b', r'\1 \2', text)

    # Convert bracketed indices like x[n], y[n], x1[n], x[k] to 'x of n'
    text = re.sub(r'\b([a-zA-Z0-9_]+)\[([^\]]+)\]', r'\1 of \2', text)
    text = re.sub(r'[\[\]]', '', text)

    # Split accidental fused CamelCase words (e.g. WhenNIs -> When N Is)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)

    # Collapse whitespace
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def extract_text_from_pdf_vision(pdf_path: str,
                                 model: str | None = None,
                                 skip_references: bool = True,
                                 dpi: int = 150,
                                 page_indices: list[int] | None = None) -> tuple[str, str]:
    """Render PDF pages as images and use a local Vision LLM via Ollama."""
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    if not is_ollama_available():
        raise RuntimeError("Ollama server is not running on http://localhost:11434. Start Ollama to use --vision mode.")

    ollama_model = model or get_default_ollama_model() or "qwen3.8:latest"
    doc = fitz.open(pdf_path)
    total_doc_pages = len(doc)
    target_indices = page_indices if page_indices is not None else list(range(total_doc_pages))
    total_to_process = len(target_indices)

    print(f"\n  Vision Model: {ollama_model} (reading {total_to_process} pages at {dpi} DPI)...")

    page_texts = []
    title = None
    prev_text = ""

    with tqdm(total=total_to_process, desc="  Vision Reading Pages", unit="page", ncols=85) as pbar:
        for i, page_idx in enumerate(target_indices):
            if not (0 <= page_idx < total_doc_pages):
                pbar.update(1)
                continue
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=dpi)
            img_b64 = base64.b64encode(pix.tobytes("jpeg")).decode("utf-8")

            messages = [
                {
                    "role": "user",
                    "content": build_vision_prompt(prev_text),
                    "images": [img_b64],
                }
            ]

            gen_text = query_ollama(messages, model=ollama_model, temperature=0.2)
            gen_text = clean_spoken_math_text(gen_text)
            prev_text = gen_text

            if gen_text.strip():
                if title is None:
                    first_line = gen_text.strip().split("\n")[0]
                    title = first_line[:120].strip()

                if skip_references and re.search(r'(?i)\b(references|bibliography)\b', gen_text) and len(gen_text.split()) < 150:
                    page_texts.append(gen_text)
                    pbar.update(total_to_process - i)
                    break

                page_texts.append(gen_text)

            pbar.update(1)

    doc.close()
    if not title:
        title = "Technical Paper"

    full_text = "\n\n".join(page_texts).strip()
    return title, full_text


# ---------------------------------------------------------------------------
# Text Preprocessing & Cleaning Pipeline
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """Clean extracted textbook text for natural, fluid TTS narration."""
    # Strip author emails and URLs
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\b(?:doi|DOI):\s*\S+', '', text)
    text = re.sub(r'\b(?:arXiv|ARXIV):\s*\S+', '', text)

    # Rejoin hyphenated line-breaks (e.g. trans- / form -> transform)
    text = re.sub(r'(\b[a-zA-Z]+)-\n([a-zA-Z]+\b)', r'\1\2', text)

    # Remove inline brackets/citations: [1], [1, 2], [1-5]
    text = re.sub(r'\[(?:\d+[–\-, ]*)+\]', '', text)
    text = re.sub(r'\((?:[A-Z][a-z]+(?:\s+et\s+al\.?)?,\s*\d{4}[a-z]?(?:;\s*)?)+\)', '', text)

    # Announce numbered section headings cleanly
    text = re.sub(
        r'(?m)^(\d+(?:\.\d+)*)\s+([A-Z][A-Za-z0-9\s,–\-]{2,60})$',
        r'\nSection \1: \2.\n',
        text
    )

    # Remove standalone line numbers or page numbers
    text = re.sub(r'(?m)^\s*\d+\s*$', '', text)

    # Clean punctuation and normalize spacing
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def split_into_chunks(text: str, max_chars: int = 480) -> list[str]:
    """Split text into sentence-aligned chunks suitable for Kokoro TTS."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []

    for p in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', p)
        cur, cur_len = [], 0
        for s in sentences:
            if not s:
                continue
            if cur_len + len(s) > max_chars and cur:
                chunks.append(" ".join(cur))
                cur, cur_len = [s], len(s)
            else:
                cur.append(s)
                cur_len += len(s)
        if cur:
            chunks.append(" ".join(cur))

    return [c.strip() for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Local Ollama Inference
# ---------------------------------------------------------------------------

def is_ollama_available(host: str = "http://localhost:11434") -> bool:
    """Check if the local Ollama server is running."""
    try:
        req = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_default_ollama_model(host: str = "http://localhost:11434") -> str | None:
    """Get the best available local model installed in Ollama."""
    try:
        req = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            models = [m['name'] for m in data.get('models', [])]
            for preferred in ["qwen3.8:latest", "qwen3.5:latest", "qwen3:14b", "qwen3:8b", "qwen3:latest", "qwen2.5:7b", "llama3.1:latest", "llama3:latest"]:
                if preferred in models:
                    return preferred
            return models[0] if models else None
    except Exception:
        return None


def query_ollama(messages: list[dict], model: str, host: str = "http://localhost:11434", temperature: float = 0.2, num_predict: int = 2048) -> str:
    """Send chat request to Ollama with natural sampling options."""
    data = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_ctx": 16384,
            "num_predict": num_predict,
        }
    }
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        content = res["message"]["content"].strip()
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        content = re.sub(r'</?think>', '', content).strip()
        return content


# ---------------------------------------------------------------------------
# TTS Audio Synthesis (Kokoro)
# ---------------------------------------------------------------------------

VOICES = {
    "af_heart":    "American Female — warm, natural (default)",
    "af_bella":    "American Female — expressive, engaging",
    "af_nicole":   "American Female — calm, soothing",
    "af_sarah":    "American Female — crisp, clear",
    "af_sky":      "American Female — bright, conversational",
    "am_adam":     "American Male — deep, resonant",
    "am_michael":  "American Male — natural, professor-like",
    "bf_emma":     "British Female — crisp, academic",
    "bf_isabella": "British Female — soft, clear",
    "bm_george":   "British Male — authoritative, formal",
    "bm_lewis":    "British Male — casual, modern",
}


def generate_audio(text: str, out_wav: str, voice: str = "af_heart", speed: float = 1.0):
    """Synthesise text to WAV using Kokoro TTS."""
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    lang = 'b' if voice.startswith('b') else 'a'
    pipeline = KPipeline(lang_code=lang)
    chunks = split_into_chunks(text)
    if not chunks:
        raise ValueError("No text provided for audio synthesis.")

    audio_parts = []
    with tqdm(total=len(chunks), desc="  Synthesising Audio", unit="chunk", ncols=85) as pbar:
        for chunk in chunks:
            for _, _, audio in pipeline(chunk, voice=voice, speed=speed):
                audio_parts.append(audio)
            pbar.update(1)

    if not audio_parts:
        raise RuntimeError("No audio was generated by Kokoro TTS.")

    full_audio = np.concatenate(audio_parts)
    sf.write(out_wav, full_audio, 24000)


# ---------------------------------------------------------------------------
# Audio Encoding (ffmpeg)
# ---------------------------------------------------------------------------

def encode_mp3(wav_path: str, mp3_path: str):
    """Convert WAV to MP3 using ffmpeg."""
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", wav_path,
        "-codec:a", "libmp3lame", "-q:a", "2",
        mp3_path
    ], check=True)


def encode_m4a(wav_path: str, m4a_path: str):
    """Convert WAV to M4A/AAC using ffmpeg."""
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", wav_path,
        "-codec:a", "aac", "-b:a", "192k",
        m4a_path
    ], check=True)


# ---------------------------------------------------------------------------
# Real-Time Per-Page Textbook Audiobook Engine
# ---------------------------------------------------------------------------

def process_book_per_page(pdf_path: str,
                          out_dir: Path,
                          model: str | None = None,
                          voice: str = "af_heart",
                          speed: float = 1.0,
                          use_vision: bool = True,
                          dpi: int = 150,
                          page_pairs: list[tuple[int, int]] | None = None,
                          skip_references: bool = True,
                          resume: bool = True):
    """
    Process textbook page-by-page: extracts text or renders with Vision LLM,
    synthesises Kokoro speech immediately per page, saves page_0001.mp3,
    and maintains an active M3U playlist.
    """
    import numpy as np
    from kokoro import KPipeline

    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    total_doc_pages = len(doc)
    target_pairs = page_pairs if page_pairs is not None else [(i + 1, i) for i in range(total_doc_pages)]

    ollama_model = model or get_default_ollama_model() or "qwen3.8:latest"

    # Initialize Kokoro TTS pipeline once
    lang = 'b' if voice.startswith('b') else 'a'
    print(f"Loading Kokoro TTS engine (lang={lang}, voice={voice}, speed={speed}x)...")
    pipeline = KPipeline(lang_code=lang)

    print(f"\n{'='*65}")
    print(f"  LectureLens — Real-Time Audiobook Generation")
    print(f"  Book/Paper : {Path(pdf_path).name}")
    print(f"  Output Dir : {out_dir}/")
    print(f"  Pages      : {len(target_pairs)} pages (Book pages: {[p for p, _ in target_pairs[:5]]}{'...' if len(target_pairs) > 5 else ''})")
    print(f"  Vision LLM : {ollama_model}")
    print(f"  Voice      : {voice} | Speed: {speed}x | Resume: {resume}")
    print(f"{'='*65}\n")

    generated_mp3s = []
    playlist_path = out_dir / "playlist.m3u"

    if resume and playlist_path.exists():
        with open(playlist_path, "r", encoding="utf-8") as pf:
            generated_mp3s = [line.strip() for line in pf if line.strip()]

    for step_i, (book_page_num, pdf_page_idx) in enumerate(target_pairs, start=1):
        if not (0 <= pdf_page_idx < total_doc_pages):
            continue

        page_stem = f"page_{book_page_num:04d}"
        txt_path = out_dir / f"{page_stem}.txt"
        mp3_path = out_dir / f"{page_stem}.mp3"

        if resume and mp3_path.exists() and mp3_path.stat().st_size > 1000:
            offset_info = f" (PDF page {pdf_page_idx + 1})" if (book_page_num != pdf_page_idx + 1) else ""
            print(f"[{step_i}/{len(target_pairs)}] Page {book_page_num}{offset_info}: Already exists ({mp3_path.name}) — Skipping (Resume).")
            if mp3_path.name not in generated_mp3s:
                generated_mp3s.append(mp3_path.name)
            continue

        offset_info = f" (PDF page {pdf_page_idx + 1})" if (book_page_num != pdf_page_idx + 1) else ""
        print(f"\n[{step_i}/{len(target_pairs)}] >>> Processing Page {book_page_num}{offset_info}...")

        # 1. Text extraction / Vision LLM
        page = doc[pdf_page_idx]
        page_text = ""

        # Retrieve previous page text context for seamless page-to-page continuity
        prev_context = ""
        prev_page_file = out_dir / f"page_{book_page_num - 1:04d}.txt"
        if prev_page_file.exists():
            try:
                with open(prev_page_file, "r", encoding="utf-8") as pf:
                    prev_context = pf.read().strip()
            except Exception:
                prev_context = ""

        if use_vision:
            pix = page.get_pixmap(dpi=dpi)
            img_b64 = base64.b64encode(pix.tobytes("jpeg")).decode("utf-8")
            messages = [
                {
                    "role": "user",
                    "content": build_vision_prompt(prev_context),
                    "images": [img_b64],
                }
            ]
            gen_text = query_ollama(messages, model=ollama_model, temperature=0.2)
            page_text = clean_spoken_math_text(gen_text)
        else:
            raw_blocks = page.get_text("blocks", sort=False)
            font_sizes = _block_font_sizes(page)
            p_blocks = [
                b + (font_sizes.get(b[5], 0),)
                for b in raw_blocks if b[6] == 0 and b[4].strip()
            ]
            sorted_blocks = _sort_blocks_two_column(p_blocks, page.rect.width)
            raw = "\n\n".join(b[4].strip() for b in sorted_blocks if b[4].strip())
            page_text = preprocess_text(raw)

        if skip_references and re.search(r'(?i)\b(references|bibliography)\b', page_text) and len(page_text.split()) < 150:
            print(f"  Page {book_page_num} identified as references section. Stopping.")
            break

        if not page_text.strip():
            print(f"  Page {book_page_num}: No readable text found. Skipping audio generation.")
            continue

        # 2. Save page transcript
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(page_text)

        # 3. Synthesize Kokoro audio for this page
        chunks = split_into_chunks(page_text)
        audio_parts = []
        for chunk in chunks:
            for _, _, audio in pipeline(chunk, voice=voice, speed=speed):
                audio_parts.append(audio)

        if audio_parts:
            full_audio = np.concatenate(audio_parts)
            duration = len(full_audio) / 24000

            # Write temporary WAV and convert to MP3
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_wav:
                tmp_wav = f_wav.name
            try:
                import soundfile as sf
                sf.write(tmp_wav, full_audio, 24000)
                encode_mp3(tmp_wav, str(mp3_path))
            finally:
                if os.path.exists(tmp_wav):
                    os.unlink(tmp_wav)

            print(f"  --> [READY FOR LISTENING] Page {book_page_num} -> {mp3_path} ({duration:.1f}s, {len(page_text.split())} words)")
            if mp3_path.name not in generated_mp3s:
                generated_mp3s.append(mp3_path.name)

            # Update M3U playlist after every completed page
            with open(playlist_path, "w", encoding="utf-8") as pf:
                for track in sorted(generated_mp3s):
                    pf.write(f"{track}\n")

    doc.close()
    print(f"\n{'='*65}")
    print(f"  All requested pages completed!")
    print(f"  Saved in folder : {out_dir}/")
    print(f"  Playlist        : {playlist_path}")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# CLI Argument Parser & Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LectureLens — Turn Textbooks and Papers into Spoken Professor Lectures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Textbook Mode (Page-by-page real-time audio with page offset):
  python lecturelens.py textbook.pdf --per-page --pages "10-61,99-135" --offset 23

  # Direct Paper Conversion (arXiv URL to MP3):
  python lecturelens.py https://arxiv.org/pdf/2506.10947.pdf --mp3 --speed 1.2

  # Custom Vision Model & Output Folder:
  python lecturelens.py book.pdf --per-page --model qwen3.8:latest --out-dir ./Audiobook/Math
"""
    )
    parser.add_argument("pdf", nargs="?", help="PDF file path or direct URL.")
    parser.add_argument("-o", "--output", help="Output audio file path.")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier (0.5–3.0, default: 1.0).")
    parser.add_argument("--voice", default="af_heart", help="Kokoro voice ID (default: af_heart). Run --list-voices.")
    parser.add_argument("--pages", default=None, metavar="RANGE", help="Page range to process (e.g. 10-61 or 10-61,99-135).")
    parser.add_argument("--offset", type=int, default=0, metavar="N", help="Offset between printed page numbers and PDF indices.")
    parser.add_argument("--max-pages", type=int, default=None, metavar="N", help="Maximum number of pages to process.")
    parser.add_argument("--no-references", action="store_true", help="Stop before References / Bibliography section.")
    parser.add_argument("--audio-only", action="store_true", help="Save output as uncompressed WAV.")
    parser.add_argument("--mp3", action="store_true", help="Save output as MP3 (default in per-page mode).")
    parser.add_argument("--m4a", action="store_true", help="Save output as M4A (Apple-native).")
    parser.add_argument("--list-voices", action="store_true", help="Print available voices and exit.")
    parser.add_argument("--vision", action="store_true", default=True, help="Use Ollama Vision LLM for page reading (default: True).")
    parser.add_argument("--dpi", type=int, default=150, help="DPI rendering resolution for Vision processing (default: 150).")
    parser.add_argument("--per-page", action="store_true", help="Process page-by-page in real time with M3U playlist.")
    parser.add_argument("--out-dir", default=None, metavar="DIR", help="Output directory for --per-page mode.")
    parser.add_argument("--no-resume", action="store_true", help="Regenerate already completed pages.")
    parser.add_argument("--model", "--ollama-model", "-m", default=None, metavar="MODEL", help="Ollama model name (e.g. qwen3.8:latest, qwen3.5:latest).")

    args = parser.parse_args()

    if args.list_voices:
        print("\nAvailable Kokoro Voices:")
        for vid, desc in VOICES.items():
            marker = " (default)" if vid == "af_heart" else ""
            print(f"  {vid:<14}  {desc}{marker}")
        print()
        return

    if not args.pdf:
        parser.print_help()
        sys.exit(1)

    is_url = args.pdf.startswith("http://") or args.pdf.startswith("https://")
    _tmp_pdf = None

    if is_url:
        url = args.pdf
        url_filename = Path(urllib.parse.urlparse(url).path).name
        url_stem = url_filename.rsplit(".", 1)[0] if "." in url_filename else url_filename or "downloaded"
        _tmp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        _tmp_pdf.close()
        try:
            download_pdf(url, _tmp_pdf.name)
        except Exception as e:
            os.unlink(_tmp_pdf.name)
            sys.exit(f"Error downloading PDF: {e}")
        pdf_path = Path(_tmp_pdf.name)
        url_stem = url_stem
    else:
        pdf_path = Path(args.pdf).expanduser().resolve()
        if not pdf_path.exists():
            sys.exit(f"Error: file not found — {pdf_path}")
        url_stem = pdf_path.stem

    if args.voice not in VOICES:
        print(f"Warning: unknown voice '{args.voice}'. Run --list-voices to see options.")

    if not (0.3 <= args.speed <= 3.0):
        sys.exit("Error: --speed must be between 0.3 and 3.0")

    # Real-Time Per-Page Mode
    if args.per_page:
        out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else Path(url_stem).resolve()
        clean_model = args.model

        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        _doc_info = fitz.open(str(pdf_path))
        total_doc_pages = len(_doc_info)
        _doc_info.close()

        page_pairs = parse_page_selection(args.pages, args.max_pages, total_doc_pages, offset=args.offset)

        try:
            process_book_per_page(
                str(pdf_path),
                out_dir=out_dir,
                model=clean_model,
                voice=args.voice,
                speed=args.speed,
                use_vision=True,
                dpi=args.dpi,
                page_pairs=page_pairs,
                skip_references=args.no_references,
                resume=not args.no_resume,
            )
        finally:
            if _tmp_pdf and os.path.exists(_tmp_pdf.name):
                os.unlink(_tmp_pdf.name)
        return

    # Monolithic Single-File Mode
    def _stem_path(ext: str) -> Path:
        if args.output:
            return Path(args.output).expanduser().resolve()
        return pdf_path.parent / f"{url_stem}{ext}"

    if args.audio_only:
        out_path = _stem_path(".wav")
    elif args.m4a:
        out_path = _stem_path(".m4a")
    else:
        out_path = _stem_path(".mp3")

    print(f"\n  LectureLens — Direct Paper Narration")
    print(f"  PDF  : {pdf_path.name}")
    print(f"  Out  : {out_path.name}")
    print(f"  Voice: {args.voice} | Speed: {args.speed}x\n")

    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    _doc_info = fitz.open(str(pdf_path))
    total_doc_pages = len(_doc_info)
    _doc_info.close()

    page_indices = [idx for _, idx in parse_page_selection(args.pages, args.max_pages, total_doc_pages, offset=args.offset)]

    title, clean_text = extract_text_from_pdf_vision(
        str(pdf_path),
        model=args.model,
        skip_references=args.no_references,
        dpi=args.dpi,
        page_indices=page_indices,
    )

    word_count = len(clean_text.split())
    est_min = word_count / (150 * args.speed)
    print(f"  {word_count:,} words  →  estimated audio length: ~{est_min:.0f} min\n")

    if args.audio_only:
        generate_audio(clean_text, str(out_path), voice=args.voice, speed=args.speed)
    elif args.m4a:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = f.name
        try:
            generate_audio(clean_text, tmp_wav, voice=args.voice, speed=args.speed)
            encode_m4a(tmp_wav, str(out_path))
        finally:
            if os.path.exists(tmp_wav):
                os.unlink(tmp_wav)
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = f.name
        try:
            generate_audio(clean_text, tmp_wav, voice=args.voice, speed=args.speed)
            encode_mp3(tmp_wav, str(out_path))
        finally:
            if os.path.exists(tmp_wav):
                os.unlink(tmp_wav)

    if _tmp_pdf and os.path.exists(_tmp_pdf.name):
        os.unlink(_tmp_pdf.name)

    print(f"\nDone!  →  {out_path}\n")


if __name__ == "__main__":
    main()
