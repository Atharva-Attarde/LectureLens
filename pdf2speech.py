#!/usr/bin/env python3
"""
Pdf2Speech — Convert scientific PDFs to audio/MP4
Uses Kokoro TTS (high-quality, runs locally on CPU/GPU)

Usage:
  python pdf2speech.py paper.pdf
  python pdf2speech.py paper.pdf --speed 1.3 --voice af_bella
  python pdf2speech.py paper.pdf --audio-only
  python pdf2speech.py paper.pdf --no-references --speed 1.5
  python pdf2speech.py paper.pdf --podcast            # direct MP3 + podcast MP3
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
# URL download
# ---------------------------------------------------------------------------

def download_pdf(url: str, dest_path: str):
    """Download a PDF from a URL to dest_path, showing a tqdm progress bar with speed and ETA."""
    print(f"Downloading PDF from:\n  {url}")

    req = urllib.request.Request(url, headers={"User-Agent": "Pdf2Speech/1.0"})
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as f:
        total_size = int(response.headers.get("Content-Length", 0))
        with tqdm(total=total_size if total_size > 0 else None,
                  unit="B", unit_scale=True, unit_divisor=1024,
                  desc="  Downloading PDF", ncols=85) as pbar:
            block_size = 65536
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
    print()


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

_CAPTION_RE = re.compile(r'^\s*(figure|fig\.|table)\s*\d', re.IGNORECASE)


def _block_font_sizes(page) -> dict:
    """Map each text block's number to its largest span font size."""
    sizes = {}
    for blk in page.get_text("dict")["blocks"]:
        if blk.get("type") != 0:
            continue
        spans = [s["size"] for line in blk["lines"] for s in line["spans"]]
        if spans:
            sizes[blk["number"]] = max(spans)
    return sizes


def _detect_column_margins(pages_blocks, page_width: float) -> tuple[float, float] | None:
    """
    Find the two dominant left-margin x0 values (one per column) shared by real
    body-text blocks across a document, by clustering x0 to the nearest 5pt bucket.
    Returns (left_x, right_x) or None if no clear two-column pattern is found.
    """
    from collections import Counter

    mid_x = page_width / 2
    left_counts, right_counts = Counter(), Counter()
    for blocks in pages_blocks:
        for b in blocks:
            x0, x1 = b[0], b[2]
            if (x1 - x0) / page_width > 0.6:
                continue
            bucket = round(x0 / 5) * 5
            if (x0 + x1) / 2 < mid_x:
                left_counts[bucket] += 1
            else:
                right_counts[bucket] += 1

    if not left_counts or not right_counts:
        return None
    return left_counts.most_common(1)[0][0], right_counts.most_common(1)[0][0]


def _sort_blocks_two_column(blocks, page_width: float, margins: tuple[float, float] | None = None) -> list:
    """
    Sort text blocks for a page that may have a two-column layout.
    Full-width blocks (title, abstract, headings) are interleaved by y-position;
    column blocks are read left-column-first, then right-column.
    Blocks that don't align to a known column margin (e.g. text labels embedded
    in figures/diagrams) are dropped unless they look like a figure/table caption.
    """
    mid_x = page_width / 2
    margin_tol = 12.0

    full_width, aligned_left, aligned_right, misaligned = [], [], [], []
    for b in blocks:
        x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
        width_fraction = (x1 - x0) / page_width
        if width_fraction > 0.6:
            full_width.append(b)
        elif (x0 + x1) / 2 < mid_x:
            if margins is None or abs(x0 - margins[0]) <= margin_tol:
                aligned_left.append(b)
            else:
                misaligned.append((b, "left"))
        else:
            if margins is None or abs(x0 - margins[1]) <= margin_tol:
                aligned_right.append(b)
            else:
                misaligned.append((b, "right"))

    # Off-margin blocks are either legit content set in body-size type (a
    # centered title, an equation, a block quote) or small-font labels
    # embedded in a figure/diagram. Font size — not position — tells them
    # apart, since diagram labels are almost always smaller than body text.
    body_sizes = [b[7] for b in aligned_left + aligned_right if len(b) > 7]
    body_size = sorted(body_sizes)[len(body_sizes) // 2] if body_sizes else 0
    for b, side in misaligned:
        target = aligned_left if side == "left" else aligned_right
        size = b[7] if len(b) > 7 else body_size
        if _CAPTION_RE.match(b[4].strip()) or size >= body_size - 0.5:
            target.append(b)

    left_col, right_col = aligned_left, aligned_right

    # If there are almost no column blocks, fall back to plain y-sort
    if len(left_col) + len(right_col) < 2:
        return sorted(blocks, key=lambda b: (b[1], b[0]))

    left_col.sort(key=lambda b: b[1])
    right_col.sort(key=lambda b: b[1])
    full_width.sort(key=lambda b: b[1])

    # Interleave full-width sentinels with column text using y-position
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
    Parse page specification (e.g. '1', '1-3', '1,3,5') or max_pages limit with an optional page offset.
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
                start_p = int(start)
                end_p = int(end)
                selected_pages.extend(range(start_p, end_p + 1))
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
    """
    Extract text from a PDF. Returns (title, body_text).
    Handles two-column academic layouts by reading left column before right.
    """
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz  # PyMuPDF

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

    title = all_blocks[0][:120].strip() if all_blocks else "Scientific Paper"
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
# Vision PDF Reading (Render pages as images for technical textbooks)
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
    """
    Render PDF pages as images and use a local Vision LLM (e.g. Qwen2.5-VL / Qwen3.5 in Ollama)
    to visually transcribe and verbalize complex math, multi-column layouts, and technical textbook pages.
    """
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    if not is_ollama_available():
        raise RuntimeError("Ollama server is not running on http://localhost:11434. Start Ollama to use --vision mode.")

    ollama_model = model or get_default_ollama_model() or "qwen3.5:latest"
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

            gen_text = query_ollama(messages, model=ollama_model, temperature=0.1)
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
        title = "Technical Textbook / Paper"

    full_text = "\n\n".join(page_texts).strip()
    return title, full_text


# ---------------------------------------------------------------------------
# Text preprocessing for TTS
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """Clean extracted PDF text so it reads naturally as speech."""

    # Fix hyphenated line-breaks (e.g. "meth-\nod" → "method")
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # Join lines that are continuation of a sentence (lowercase start)
    text = re.sub(r'\n(?=[a-z,;])', ' ', text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove numeric citations [1], [2,3], [1–4]
    text = re.sub(r'\[\d[\d,\s\-–]*\]', '', text)
    # Author-year citations  [Smith et al., 2020]
    text = re.sub(r'\[[A-Z][^]]{1,40}\d{4}\]', '', text)

    # LaTeX inline/display math → spoken placeholder
    text = re.sub(r'\$\$[^$]+\$\$', ' equation ', text, flags=re.DOTALL)
    text = re.sub(r'\$[^$\n]+\$', ' expression ', text)

    # Remove figure / table / equation cross-references
    text = re.sub(r'(?:Fig(?:ure)?|Table|Eq(?:uation)?|Algorithm)\.?\s*\d+', '', text, flags=re.IGNORECASE)

    # Remove lone page numbers
    text = re.sub(r'(?m)^\d{1,4}$', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)

    # Collapse excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # Announce section headings (ALL CAPS or Title Case on their own line)
    text = re.sub(
        r'(?m)^([A-Z][A-Z\s]{3,50})$',
        lambda m: f'\n\nSection: {m.group(1).title()}.\n\n',
        text
    )

    return text.strip()


def split_into_chunks(text: str, max_chars: int = 400) -> list[str]:
    """
    Split text into sentence-aligned chunks suitable for TTS inference.
    Kokoro works best with chunks under ~500 chars.
    """
    # Split at sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks, current, cur_len = [], [], 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if cur_len + len(sent) > max_chars and current:
            chunks.append(' '.join(current))
            current, cur_len = [sent], len(sent)
        else:
            current.append(sent)
            cur_len += len(sent)

    if current:
        chunks.append(' '.join(current))

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# TTS — Kokoro
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Google Drive upload
# ---------------------------------------------------------------------------

GDRIVE_FOLDER_ID = "1VdPiHKs4U0MfcsnwMeX0BGt-d9vqdUhZ"
_GDRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]
_GDRIVE_TOKEN_PATH = Path(__file__).parent / ".gdrive_token.json"
_GDRIVE_CREDS_PATH = Path(__file__).parent / "gdrive_credentials.json"


def _gdrive_service():
    """Return an authenticated Google Drive service, running OAuth flow if needed."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if _GDRIVE_TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(_GDRIVE_TOKEN_PATH), _GDRIVE_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not _GDRIVE_CREDS_PATH.exists():
                raise FileNotFoundError(
                    f"Google Drive credentials not found at {_GDRIVE_CREDS_PATH}\n"
                    "Download OAuth 2.0 credentials from Google Cloud Console and save as gdrive_credentials.json"
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(_GDRIVE_CREDS_PATH), _GDRIVE_SCOPES)
            creds = flow.run_local_server(port=0)
        _GDRIVE_TOKEN_PATH.write_text(creds.to_json())

    return build("drive", "v3", credentials=creds)


def upload_to_gdrive(file_path: str, folder_id: str = GDRIVE_FOLDER_ID) -> str:
    """Upload a file to Google Drive and return its shareable URL."""
    from googleapiclient.http import MediaFileUpload
    import mimetypes

    path = Path(file_path)
    mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"

    print(f"\nUploading to Google Drive...")
    service = _gdrive_service()

    file_metadata = {"name": path.name, "parents": [folder_id]}
    media = MediaFileUpload(str(path), mimetype=mime_type, resumable=True)

    uploaded = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id, name, webViewLink",
    ).execute()

    url = uploaded.get("webViewLink", f"https://drive.google.com/file/d/{uploaded['id']}/view")
    print(f"Uploaded: {uploaded['name']}")
    print(f"  → {url}")
    return url


VOICES = {
    # American English
    'af_heart':   'American Female — warm (default)',
    'af_bella':   'American Female — expressive',
    'af_nicole':  'American Female — calm',
    'af_sarah':   'American Female — clear',
    'af_sky':     'American Female — bright',
    'am_adam':    'American Male — deep',
    'am_michael': 'American Male — natural',
    # British English
    'bf_emma':    'British Female — crisp',
    'bf_isabella':'British Female — soft',
    'bm_george':  'British Male — authoritative',
    'bm_lewis':   'British Male — casual',
}


def generate_audio(text: str, out_wav: str, voice: str = 'af_heart', speed: float = 1.0):
    """
    Synthesise speech with Kokoro TTS and write a WAV file.
    speed: 0.5 = half speed, 2.0 = double speed
    """
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    # lang_code: 'a' = American English, 'b' = British English
    lang = 'b' if voice.startswith('b') else 'a'
    print(f"Loading Kokoro TTS (lang={lang}, voice={voice}, speed={speed}x)...")
    pipeline = KPipeline(lang_code=lang)

    chunks = split_into_chunks(text)
    total = len(chunks)
    audio_parts = []

    with tqdm(total=total, desc="  Synthesising Audio", unit="chunk", ncols=85) as pbar:
        for chunk in chunks:
            for _, _, audio in pipeline(chunk, voice=voice, speed=speed):
                audio_parts.append(audio)
            pbar.update(1)

    if not audio_parts:
        raise RuntimeError("No audio generated — check that the text is non-empty.")

    full_audio = np.concatenate(audio_parts)
    sf.write(out_wav, full_audio, samplerate=24000)
    duration = len(full_audio) / 24000
    print(f"Audio: {duration/60:.1f} min  →  {out_wav}")


# ---------------------------------------------------------------------------
# MP4 creation (audio + static cover image)
# ---------------------------------------------------------------------------

FFMPEG_PATHS = [
    # Try pactenv first, then known locations
    "/mnt/lustre/work/kuehne/kqr828/.conda/pactenv/bin/ffmpeg",
    "/mnt/lustre/work/kuehne/kqr828/.conda/eduvid/bin/ffmpeg",
    "ffmpeg",
]


def find_ffmpeg() -> str:
    for path in FFMPEG_PATHS:
        try:
            subprocess.run([path, "-version"], capture_output=True, check=True)
            return path
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    raise RuntimeError(
        "ffmpeg not found. Install it with:\n"
        "  conda install -n pactenv -c conda-forge ffmpeg\n"
        "or point FFMPEG_PATH env var to the binary."
    )


def make_cover_image(title: str, tmp_path: str):
    """Create a simple 1920×1080 cover image with PIL."""
    from PIL import Image, ImageDraw
    import textwrap

    W, H = 1920, 1080
    img = Image.new("RGB", (W, H), color=(12, 12, 28))
    draw = ImageDraw.Draw(img)

    # Background gradient effect (simple rectangles)
    for i in range(0, H, 4):
        shade = int(12 + 10 * (i / H))
        draw.rectangle([(0, i), (W, i + 4)], fill=(shade, shade, shade + 16))

    # Title lines
    title_clean = re.sub(r'[^\w\s\-:,.()\[\]]', '', title)
    lines = textwrap.wrap(title_clean, width=55)[:5]

    line_h = 60
    y_start = H // 2 - (len(lines) * line_h) // 2 - 40

    for j, line in enumerate(lines):
        draw.text((W // 2, y_start + j * line_h), line,
                  fill=(210, 210, 255), anchor="mm")

    draw.text((W // 2, H // 2 + 120), "Pdf2Speech — audio paper",
              fill=(90, 90, 180), anchor="mm")

    img.save(tmp_path)


def create_mp4(audio_wav: str, output_mp4: str, title: str):
    """Combine audio + cover image into an MP4 using ffmpeg."""
    ffmpeg = find_ffmpeg()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name

    try:
        try:
            make_cover_image(title, img_path)
            video_input = ["-loop", "1", "-i", img_path]
        except ImportError:
            # Pillow not available — use ffmpeg solid color
            img_path = None
            video_input = ["-f", "lavfi", "-i", "color=c=0x0C0C1C:size=1920x1080:rate=1"]

        # Try libx264 first; fall back to mpeg4 for environments without x264
        def _build_cmd(vcodec_args):
            return [
                ffmpeg, "-y",
                *video_input,
                "-i", audio_wav,
                *vcodec_args,
                "-c:a", "aac", "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-shortest",
                output_mp4,
            ]

        cmd = _build_cmd(["-c:v", "libx264", "-tune", "stillimage", "-preset", "fast"])
        probe = subprocess.run(cmd, capture_output=True, text=True)
        if probe.returncode != 0 and "libx264" in probe.stderr or "preset" in probe.stderr:
            # libx264 not available — use built-in mpeg4
            cmd = _build_cmd(["-c:v", "mpeg4", "-q:v", "5"])
        print("Encoding MP4...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("ffmpeg stderr:\n", result.stderr[-2000:])
            raise RuntimeError("ffmpeg failed — see above.")
        print(f"MP4: {output_mp4}")
    finally:
        if img_path and os.path.exists(img_path):
            os.unlink(img_path)


# ---------------------------------------------------------------------------
# Audio export helpers
# ---------------------------------------------------------------------------

def wav_to_mp3(wav_path: str, out_path: str, bitrate: str = "192k"):
    """Convert a WAV file to MP3 using ffmpeg."""
    ffmpeg = find_ffmpeg()
    cmd = [ffmpeg, "-y", "-i", wav_path, "-c:a", "libmp3lame", "-b:a", bitrate, out_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg stderr:\n", result.stderr[-2000:])
        raise RuntimeError("ffmpeg MP3 conversion failed — see above.")
    print(f"MP3: {out_path}")


def wav_to_m4a(wav_path: str, out_path: str, bitrate: str = "192k"):
    """Convert a WAV file to M4A (AAC) using ffmpeg — best for macOS/iOS."""
    ffmpeg = find_ffmpeg()
    cmd = [ffmpeg, "-y", "-i", wav_path, "-c:a", "aac", "-b:a", bitrate, out_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg stderr:\n", result.stderr[-2000:])
        raise RuntimeError("ffmpeg M4A conversion failed — see above.")
    print(f"M4A: {out_path}")


# ---------------------------------------------------------------------------
# LLM Clean: Natural Spoken Narration Preprocessing
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Ollama Local Quantized LLM Client
# ---------------------------------------------------------------------------

def is_ollama_available(host: str = "http://localhost:11434") -> bool:
    """Check if local Ollama server is running and reachable."""
    try:
        req = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_default_ollama_model(host: str = "http://localhost:11434") -> str | None:
    """Get the best available local quantized model installed in Ollama."""
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
        # Strip thinking tags / internal thoughts if present
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        content = re.sub(r'</?think>', '', content).strip()
        return content


# ---------------------------------------------------------------------------
# LLM Clean: Natural Spoken Narration Preprocessing
# ---------------------------------------------------------------------------

_LLM_CLEAN_SYSTEM = (
    "You are an expert audio narrator and science communicator preparing an academic paper "
    "to be read aloud as a complete, clear, and natural scientific audiobook. "
    "Do NOT use thinking tags, internal thoughts, or commentary. Output the cleaned text directly."
)

_LLM_CLEAN_PROMPT = """\
Convert the following excerpt from a research paper into smooth, natural spoken English suitable for text-to-speech narration.

STRICT INSTRUCTIONS:
1. PRESERVE FULL SUBSTANCE: Do NOT summarize, shorten, or omit any explanations, findings, equations, or arguments. Keep all technical substance intact.
2. REMOVE ARTIFACTS: Completely eliminate table fragments, isolated numeric lists, chart axis numbers (e.g. "0.5 0.4 0.3"), figure captions/labels ("Figure 1: ..."), raw URLs, email addresses, and PDF line-break glitches.
3. SPOKEN MATH & SYMBOLS: Translate all mathematical expressions, equations, and Greek symbols into natural spoken English (e.g., "$x \\in \\mathbb{R}$" -> "x in real numbers", "$\\gamma = 0.5$" -> "gamma equals zero point five", "$\\sum_{i=1}^n$" -> "the sum from i equals 1 to n").
4. NATURAL SPEECH CADENCE: Expand unfamiliar abbreviations where appropriate, write out numbers smoothly, and ensure sentences flow seamlessly.
5. CLEAN OUTPUT ONLY: Return ONLY the cleaned spoken text. Do NOT add preamble (like "Here is the cleaned text:"), thinking tokens (<think>), explanations, markdown formatting (no asterisks, bolding, or code fences), or conversational commentary.

Paper excerpt:
{text}

Spoken narration text:"""


def llm_clean_text_for_speech(text: str,
                              model_name: str | None = None,
                              use_hf: bool = False,
                              chunk_chars: int = 4000) -> str:
    """
    Use an LLM to clean paper text for natural, fluent spoken narration,
    eliminating table/axis noise and verbalizing math without summarizing.
    Guarantees strict chunk sizing (<4000 chars / ~1000 tokens) within a 16k context window.
    """
    # 1. Break into safe units (splits oversized paragraphs by sentence)
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    units = []
    for p in raw_paragraphs:
        if len(p) <= chunk_chars:
            units.append(p)
        else:
            sents = re.split(r'(?<=[.!?])\s+', p)
            current_unit, current_u_len = [], 0
            for s in sents:
                if current_u_len + len(s) > chunk_chars and current_unit:
                    units.append(' '.join(current_unit))
                    current_unit = [s]
                    current_u_len = len(s)
                else:
                    current_unit.append(s)
                    current_u_len += len(s)
            if current_unit:
                units.append(' '.join(current_unit))

    # 2. Group units into ~chunk_chars batches
    batches = []
    current_batch = []
    current_len = 0

    for u in units:
        if current_len + len(u) > chunk_chars and current_batch:
            batches.append("\n\n".join(current_batch))
            current_batch = [u]
            current_len = len(u)
        else:
            current_batch.append(u)
            current_len += len(u)
    if current_batch:
        batches.append("\n\n".join(current_batch))

    total_batches = len(batches)
    cleaned_segments = []

    # 1. Try Ollama first (quantized, fast, minimal VRAM)
    if not use_hf and is_ollama_available():
        ollama_model = model_name or get_default_ollama_model() or "qwen3.5:latest"
        print(f"\n  LLM Cleaner: Ollama local quantized model ({ollama_model})")
        print(f"  Cleaning {total_batches} text sections with Ollama...")

        with tqdm(total=total_batches, desc="  LLM Cleaning Narration", unit="section", ncols=85) as pbar:
            for batch in batches:
                messages = [
                    {"role": "system", "content": _LLM_CLEAN_SYSTEM},
                    {"role": "user", "content": _LLM_CLEAN_PROMPT.replace("{text}", batch)},
                ]
                gen_text = query_ollama(messages, model=ollama_model, temperature=0.3)

                # Strip markdown fences or extra asterisks if LLM produced them
                gen_text = re.sub(r'^```[\w]*\n?', '', gen_text)
                gen_text = re.sub(r'\n?```$', '', gen_text)
                gen_text = re.sub(r'[*_]{1,3}', '', gen_text)
                cleaned_segments.append(gen_text)
                pbar.update(1)

        print(f"  LLM cleaning complete across all {total_batches} sections.\n")
        return "\n\n".join(cleaned_segments).strip()

    # 2. Fallback to HuggingFace Transformers
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_model_name = model_name or "Qwen/Qwen2.5-7B-Instruct"
    print(f"\n  LLM Cleaner Model (HuggingFace): {hf_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            dtype=dtype,
            device_map="auto",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            torch_dtype=dtype,
            device_map="auto",
        )

    print(f"  Cleaning {total_batches} text sections with HuggingFace model...")
    with tqdm(total=total_batches, desc="  LLM Cleaning Narration", unit="section", ncols=85) as pbar:
        for batch in batches:
            messages = [
                {"role": "system", "content": _LLM_CLEAN_SYSTEM},
                {"role": "user", "content": _LLM_CLEAN_PROMPT.replace("{text}", batch)},
            ]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max(1024, int(len(batch) / 2)),
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            gen_text = tokenizer.decode(
                output_ids[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            ).strip()

            gen_text = re.sub(r'^```[\w]*\n?', '', gen_text)
            gen_text = re.sub(r'\n?```$', '', gen_text)
            gen_text = re.sub(r'[*_]{1,3}', '', gen_text)
            cleaned_segments.append(gen_text)
            pbar.update(1)

    print(f"  LLM cleaning complete across all {total_batches} sections.\n")

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return "\n\n".join(cleaned_segments).strip()


# ---------------------------------------------------------------------------
# Podcast: LLM script generation + two-voice synthesis
# ---------------------------------------------------------------------------

_PODCAST_SYSTEM = (
    "You are an expert podcast script writer for science communication. "
    "Your podcasts make complex research accessible and engaging to a broad audience."
)

_PODCAST_PROMPT = """\
Create an engaging podcast episode about this scientific paper.
Write a natural conversation between HOST (a curious science journalist) \
and GUEST (the paper's lead researcher).
Cover the key ideas, methodology, surprising findings, and real-world implications.

STRICT FORMAT — every line must be exactly one of:
HOST: <text>
GUEST: <text>

Rules:
- 13 to 15 exchanges total, alternating HOST then GUEST
- Each turn: 2–4 sentences, conversational and jargon-free
- No markdown, asterisks, stage directions, or blank lines between turns
- Open with HOST welcoming the audience and introducing the paper topic

Paper content:
{text}

Begin the transcript now:"""


def generate_podcast_script(text: str,
                             model_name: str | None = None,
                             use_hf: bool = False) -> str:
    """Use Ollama or a Qwen instruct model to convert paper text into a HOST/GUEST podcast script."""
    paper_excerpt = text[:12000]
    if len(text) > 12000:
        paper_excerpt += "\n\n[... paper continues ...]"

    messages = [
        {"role": "system", "content": _PODCAST_SYSTEM},
        {"role": "user",   "content": _PODCAST_PROMPT.replace("{text}", paper_excerpt)},
    ]

    # 1. Try Ollama first
    if not use_hf and is_ollama_available():
        ollama_model = model_name or get_default_ollama_model() or "qwen3.5:latest"
        print(f"\n  Podcast LLM: Ollama local quantized model ({ollama_model})")
        print("  Generating podcast script...")
        return query_ollama(messages, model=ollama_model, temperature=0.7)

    # 2. Fallback to HuggingFace
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_model = model_name or "Qwen/Qwen2.5-7B-Instruct"
    print(f"\n  Podcast Model (HuggingFace): {hf_model}")
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_model,
            dtype=dtype,
            device_map="auto",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            hf_model,
            torch_dtype=dtype,
            device_map="auto",
        )

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    print("  Generating podcast script (this may take a minute)...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return generated.strip()


def parse_podcast_script(script: str) -> list[tuple[str, str]]:
    """Parse HOST:/GUEST: lines into (speaker, text) tuples."""
    turns = []
    for line in script.splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("HOST:"):
            txt = line[5:].strip()
            if txt:
                turns.append(("HOST", txt))
        elif upper.startswith("GUEST:"):
            txt = line[6:].strip()
            if txt:
                turns.append(("GUEST", txt))
    return turns


def synthesize_podcast(script: str, out_mp3: str,
                        host_voice: str = "af_bella",
                        guest_voice: str = "am_michael",
                        speed: float = 1.0):
    """Synthesise a two-voice podcast from a HOST/GUEST script and save as MP3."""
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    turns = parse_podcast_script(script)
    if not turns:
        raise RuntimeError(
            "No HOST:/GUEST: lines found in the generated script.\n"
            "Raw output:\n" + script[:800]
        )

    print(f"  {len(turns)} turns  (host={host_voice}, guest={guest_voice})")

    # One KPipeline per lang_code (American vs British)
    _pipe_cache: dict[str, KPipeline] = {}

    def _pipeline(voice: str) -> KPipeline:
        lang = "b" if voice.startswith("b") else "a"
        if lang not in _pipe_cache:
            _pipe_cache[lang] = KPipeline(lang_code=lang)
        return _pipe_cache[lang]

    silence = np.zeros(int(0.45 * 24000), dtype=np.float32)
    audio_parts: list[np.ndarray] = []

    with tqdm(total=len(turns), desc="  Synthesising Podcast", unit="turn", ncols=85) as pbar:
        for speaker, text in turns:
            voice = host_voice if speaker == "HOST" else guest_voice
            turn_chunks = [chunk for _, _, chunk in _pipeline(voice)(text, voice=voice, speed=speed)]
            if turn_chunks:
                audio_parts.append(np.concatenate(turn_chunks))
                audio_parts.append(silence)
            pbar.update(1)

    if not audio_parts:
        raise RuntimeError("No audio synthesised for podcast.")

    full_audio = np.concatenate(audio_parts)
    duration = len(full_audio) / 24000
    print(f"  Podcast audio: {duration / 60:.1f} min")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name
    try:
        sf.write(tmp_wav, full_audio, samplerate=24000)
        wav_to_mp3(tmp_wav, out_mp3)
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)



def process_book_per_page(
    pdf_path: str,
    out_dir: Path,
    model: str | None = None,
    voice: str = "af_heart",
    speed: float = 1.0,
    use_vision: bool = True,
    use_llm_clean: bool = False,
    dpi: int = 150,
    page_pairs: list[tuple[int, int]] | None = None,
    resume: bool = True,
    skip_references: bool = True,
):
    """
    Process a PDF/textbook page-by-page. For each page:
      1. Transcribes/cleans the page text using Vision LLM or text LLM.
      2. Saves page text transcript (e.g. page_0001.txt).
      3. Generates the page audio file (e.g. page_0001.mp3) IMMEDIATELY so listening can start in real time.
      4. Maintains an updated playlist.m3u playlist in the output directory.
    """
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    total_doc_pages = len(doc)
    target_pairs = page_pairs if page_pairs is not None else [(i + 1, i) for i in range(total_doc_pages)]

    ollama_model = model or get_default_ollama_model() or "qwen3.5:latest"

    # Initialize Kokoro TTS pipeline once
    lang = 'b' if voice.startswith('b') else 'a'
    print(f"Loading Kokoro TTS engine (lang={lang}, voice={voice}, speed={speed}x)...")
    pipeline = KPipeline(lang_code=lang)

    print(f"\n{'='*60}")
    print(f"  Book/Paper : {Path(pdf_path).name}")
    print(f"  Output Dir : {out_dir}/")
    print(f"  Mode       : Real-time Per-Page Processing ({'Vision LLM' if use_vision else 'Text Extraction'})")
    print(f"  Pages      : {len(target_pairs)} pages (Book pages: {[p for p, _ in target_pairs]})")
    print(f"  Model      : {ollama_model}")
    print(f"  Voice      : {voice} | Speed: {speed}x | Resume: {resume}")
    print(f"{'='*60}\n")

    generated_mp3s = []
    playlist_path = out_dir / "playlist.m3u"

    # Pre-populate playlist if resuming
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
            if use_llm_clean:
                page_text = llm_clean_text_for_speech(raw, model_name=ollama_model)
            else:
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
                sf.write(tmp_wav, full_audio, samplerate=24000)
                wav_to_mp3(tmp_wav, str(mp3_path))
            finally:
                if os.path.exists(tmp_wav):
                    os.unlink(tmp_wav)

            if mp3_path.name not in generated_mp3s:
                generated_mp3s.append(mp3_path.name)

            # Update playlist.m3u immediately
            with open(playlist_path, "w", encoding="utf-8") as pf:
                for mp3_name in generated_mp3s:
                    pf.write(f"{mp3_name}\n")

            print(f"  --> [READY FOR LISTENING] Page {book_page_num} -> {mp3_path} ({duration:.1f}s, {len(page_text.split())} words)")

    doc.close()
    print(f"\n{'='*60}")
    print(f"  All requested pages completed!")
    print(f"  Saved in folder : {out_dir}/")
    print(f"  Playlist        : {playlist_path}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert a scientific PDF to an MP4 audio file (listen like a podcast).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Default — generate MP4 at 1× speed
  python pdf2speech.py paper.pdf

  # 30% faster, British male voice
  python pdf2speech.py paper.pdf --speed 1.3 --voice bm_george

  # Skip references, output only WAV (no video)
  python pdf2speech.py paper.pdf --no-references --audio-only

  # Custom output name
  python pdf2speech.py paper.pdf -o ~/Desktop/paper_audio.mp4

  # List available voices
  python pdf2speech.py --list-voices

  # Download directly from a URL
  python pdf2speech.py https://arxiv.org/pdf/2506.10947.pdf

  # Save as MP3 (great for Mac / phones)
  python pdf2speech.py paper.pdf --mp3

  # Save as M4A (native Apple format, plays in QuickTime)
  python pdf2speech.py paper.pdf --m4a

  # Generate direct MP3 + AI podcast MP3 (Q&A format, two voices)
  python pdf2speech.py paper.pdf --podcast
  python pdf2speech.py https://arxiv.org/pdf/2506.10947.pdf --podcast --speed 1.2

  # Real-time Per-Page Audiobook (start listening to Page 1 immediately!)
  python pdf2speech.py textbook.pdf --per-page --pages 1-10 --speed 1.3
        """
    )
    parser.add_argument("pdf", nargs="?", help="Input PDF file path or URL (http/https)")
    parser.add_argument("-o", "--output", help="Output file (default: <pdf_stem>.mp4 or .wav)")
    parser.add_argument(
        "--speed", type=float, default=1.0, metavar="X",
        help="Speech speed multiplier (0.5–2.5, default: 1.0). Try 1.2–1.4 for comfortable listening."
    )
    parser.add_argument(
        "--voice", default="af_heart",
        help="Kokoro voice ID (default: af_heart). See --list-voices."
    )
    parser.add_argument(
        "--pages", default=None, metavar="RANGE",
        help="Specific pages to process, e.g. '1', '1-5', or '1,3,5' (1-indexed)."
    )
    parser.add_argument(
        "--offset", "--page-offset", type=int, default=0, metavar="N",
        help="Page number offset to add to requested page numbers to match PDF page index (e.g. --offset 16 if printed page 1 is PDF page 17)."
    )
    parser.add_argument(
        "--max-pages", type=int, default=None, metavar="N",
        help="Process only the first N pages."
    )
    parser.add_argument(
        "--no-references", action="store_true",
        help="Stop before the References / Bibliography section."
    )
    parser.add_argument(
        "--audio-only", action="store_true",
        help="Save WAV audio only; skip MP4 encoding."
    )
    parser.add_argument(
        "--mp3", action="store_true",
        help="Save as MP3 audio (universally compatible, recommended for Mac/mobile)."
    )
    parser.add_argument(
        "--m4a", action="store_true",
        help="Save as M4A/AAC audio (native Apple format, plays in QuickTime/iTunes)."
    )
    parser.add_argument(
        "--list-voices", action="store_true",
        help="Print available voices and exit."
    )
    # LLM Narration Cleaning mode
    parser.add_argument(
        "--llm-clean", action="store_true",
        help="Use an LLM (local Ollama or HF) to clean paper text for natural spoken narration (removes table/axis noise, verbalizes math without summarizing)."
    )
    parser.add_argument(
        "--vision", action="store_true",
        help="Use a Vision LLM (via local Ollama) to render PDF pages as images (ideal for technical textbooks, complex math, sidebars, and multi-column figures)."
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Rendering resolution (DPI) for --vision page processing (default: 150)."
    )
    parser.add_argument(
        "--per-page", action="store_true",
        help="Process book page-by-page in real time: saves page_0001.mp3, page_0002.mp3, etc. into a dedicated folder so you can start listening immediately."
    )
    parser.add_argument(
        "--out-dir", default=None, metavar="DIR",
        help="Output directory for --per-page mode (default: ./<book_name>/)."
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Overwrite existing page files in --per-page mode instead of skipping completed pages."
    )
    parser.add_argument(
        "--ollama-model", "--model", "-m", default=None, metavar="MODEL",
        help="Ollama model name to use (default: auto-detected, e.g. qwen3.8:latest, qwen3.5:latest, qwen3:8b, etc.)."
    )
    parser.add_argument(
        "--clean-model", default=None, metavar="MODEL",
        help="Model name for narration cleaning (Ollama model or HuggingFace path)."
    )
    parser.add_argument(
        "--use-hf", action="store_true",
        help="Force using HuggingFace Transformers instead of local Ollama."
    )
    parser.add_argument(
        "--save-cleaned-text", action="store_true",
        help="Save the preprocessed/LLM-cleaned narration text to a .txt file."
    )
    # Podcast mode
    parser.add_argument(
        "--podcast", action="store_true",
        help=(
            "Generate TWO MP3 files: a direct transcription (<stem>.mp3) "
            "and a Q&A podcast version (<stem>_podcast.mp3) written by an LLM."
        ),
    )
    parser.add_argument(
        "--podcast-model", default=None, metavar="MODEL",
        help="Model for podcast script generation (Ollama model or HuggingFace path).",
    )
    parser.add_argument(
        "--host-voice", default="af_bella", metavar="VOICE",
        help="Kokoro voice for the podcast HOST (default: af_bella).",
    )
    parser.add_argument(
        "--guest-voice", default="am_michael", metavar="VOICE",
        help="Kokoro voice for the podcast GUEST (default: am_michael).",
    )
    parser.add_argument(
        "--no-upload", action="store_true",
        help="Skip automatic Google Drive upload after generation.",
    )
    parser.add_argument(
        "--gdrive-folder", default=GDRIVE_FOLDER_ID, metavar="FOLDER_ID",
        help=f"Google Drive folder ID to upload to (default: {GDRIVE_FOLDER_ID}).",
    )

    args = parser.parse_args()

    if args.list_voices:
        print("\nAvailable Kokoro voices:")
        for vid, desc in VOICES.items():
            marker = " (default)" if vid == "af_heart" else ""
            print(f"  {vid:<14}  {desc}{marker}")
        print()
        return

    if not args.pdf:
        parser.print_help()
        sys.exit(1)

    is_url = args.pdf.startswith("http://") or args.pdf.startswith("https://")
    _tmp_pdf = None  # holds NamedTemporaryFile when downloading

    if is_url:
        url = args.pdf
        # Derive a stem from the URL filename (e.g. "2506.10947")
        url_filename = Path(urllib.parse.urlparse(url).path).name  # e.g. "2506.10947.pdf"
        url_stem = url_filename.rsplit(".", 1)[0] if "." in url_filename else url_filename or "downloaded"
        _tmp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        _tmp_pdf.close()
        try:
            download_pdf(url, _tmp_pdf.name)
        except Exception as e:
            os.unlink(_tmp_pdf.name)
            sys.exit(f"Error downloading PDF: {e}")
        pdf_path = Path(_tmp_pdf.name)
        pdf_display_name = url_stem + ".pdf"
    else:
        pdf_path = Path(args.pdf).expanduser().resolve()
        if not pdf_path.exists():
            sys.exit(f"Error: file not found — {pdf_path}")
        pdf_display_name = pdf_path.name
        url_stem = pdf_path.stem

    if args.voice not in VOICES:
        print(f"Warning: unknown voice '{args.voice}'. Run --list-voices to see options.")

    if not (0.3 <= args.speed <= 3.0):
        sys.exit("Error: --speed must be between 0.3 and 3.0")

    # Real-time Per-Page Textbook Mode
    if args.per_page:
        out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else Path(url_stem).resolve()
        clean_model = args.ollama_model or args.clean_model

        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        _doc_info = fitz.open(str(pdf_path))
        total_doc_pages = len(_doc_info)
        _doc_info.close()

        page_pairs = parse_page_selection(args.pages, args.max_pages, total_doc_pages, offset=args.offset)
        use_vision = args.vision or (not args.llm_clean)

        try:
            process_book_per_page(
                str(pdf_path),
                out_dir=out_dir,
                model=clean_model,
                voice=args.voice,
                speed=args.speed,
                use_vision=use_vision,
                use_llm_clean=args.llm_clean,
                dpi=args.dpi,
                page_pairs=page_pairs,
                resume=not args.no_resume,
                skip_references=args.no_references,
            )
        finally:
            if _tmp_pdf and os.path.exists(_tmp_pdf.name):
                os.unlink(_tmp_pdf.name)
        return

    # Determine output format and path
    if sum([args.audio_only, args.mp3, args.m4a]) > 1:
        sys.exit("Error: --audio-only, --mp3, and --m4a are mutually exclusive.")

    def _stem_path(suffix):
        # suffix may be ".mp3" or "_podcast.mp3" — always append to stem
        if is_url:
            return Path(url_stem + suffix)
        return pdf_path.parent / (pdf_path.stem + suffix)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    elif args.mp3 or args.podcast:
        out_path = _stem_path(".mp3")
    elif args.m4a:
        out_path = _stem_path(".m4a")
    elif args.audio_only:
        out_path = _stem_path(".wav")
    else:
        out_path = _stem_path(".mp4")

    podcast_out = _stem_path("_podcast.mp3") if args.podcast else None

    print(f"\n{'='*60}")
    print(f"  PDF  : {pdf_display_name}")
    if args.podcast:
        print(f"  Out  : {out_path.name}  +  {podcast_out.name}")
        print(f"  Mode : direct transcription  +  AI podcast (Q&A)")
    elif args.vision:
        print(f"  Out  : {out_path.name}")
        print(f"  Mode : Vision-based page reading (rendering pages as images for technical textbooks)")
    elif args.llm_clean:
        print(f"  Out  : {out_path.name}")
        print(f"  Mode : LLM-cleaned natural narration (full paper, zero nonsense)")
    else:
        print(f"  Out  : {out_path.name}")
    print(f"  Voice: {args.voice}  |  Speed: {args.speed}x")
    print(f"  Skip references: {args.no_references}")
    print(f"{'='*60}\n")

    # 1. Extract & Preprocess
    clean_model = args.ollama_model or args.clean_model

    # Determine page selection
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    _doc_info = fitz.open(str(pdf_path))
    total_doc_pages = len(_doc_info)
    _doc_info.close()

    page_pairs = parse_page_selection(args.pages, args.max_pages, total_doc_pages, offset=args.offset)
    page_indices = [idx for _, idx in page_pairs]
    if args.pages or args.max_pages or args.offset:
        print(f"  Pages: Processing {len(page_pairs)} pages (Book pages: {[p for p, _ in page_pairs]}, PDF indices: {[idx + 1 for _, idx in page_pairs]})\n")

    if args.vision:
        print("Step 1/2  Reading PDF via Vision LLM (rendering pages as images)...")
        title, clean_text = extract_text_from_pdf_vision(
            str(pdf_path),
            model=clean_model,
            skip_references=args.no_references,
            dpi=args.dpi,
            page_indices=page_indices,
        )
        print(f"  Title : {title[:70]}")
        print(f"  Length: {len(clean_text):,} chars  (~{len(clean_text.split()):,} words)\n")
    else:
        print("Step 1/3  Extracting text from PDF...")
        title, raw_text = extract_text_from_pdf(
            str(pdf_path),
            skip_references=args.no_references,
            page_indices=page_indices,
        )
        print(f"  Title : {title[:70]}")
        print(f"  Length: {len(raw_text):,} chars  (~{len(raw_text.split()):,} words)\n")

        print("Step 2/3  Preprocessing text...")
        if args.llm_clean:
            clean_text = llm_clean_text_for_speech(raw_text, model_name=clean_model, use_hf=args.use_hf)
        else:
            clean_text = preprocess_text(raw_text)

    if args.save_cleaned_text:
        txt_out = _stem_path("_cleaned.txt")
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(clean_text)
        print(f"  Saved cleaned narration transcript → {txt_out.name}")

    word_count = len(clean_text.split())
    est_min = word_count / (150 * args.speed)
    print(f"  {word_count:,} words  →  estimated audio length: ~{est_min:.0f} min\n")

    # Step: TTS Speech Synthesis
    step_num = "Step 2/2" if args.vision else "Step 3/3"
    print(f"{step_num}  Synthesising speech...")
    if args.audio_only:
        generate_audio(clean_text, str(out_path), voice=args.voice, speed=args.speed)
    elif args.mp3 or args.m4a or args.podcast:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = f.name
        try:
            generate_audio(clean_text, tmp_wav, voice=args.voice, speed=args.speed)
            print()
            if args.m4a:
                wav_to_m4a(tmp_wav, str(out_path))
            else:
                wav_to_mp3(tmp_wav, str(out_path))
        finally:
            if os.path.exists(tmp_wav):
                os.unlink(tmp_wav)
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = f.name
        try:
            generate_audio(clean_text, tmp_wav, voice=args.voice, speed=args.speed)
            print()
            create_mp4(tmp_wav, str(out_path), title=title)
        finally:
            if os.path.exists(tmp_wav):
                os.unlink(tmp_wav)

    print(f"\n  Direct transcription  →  {out_path}" if args.podcast else f"\nDone!  →  {out_path}")

    # Upload direct output (skip WAV — upload MP3/M4A/MP4 only)
    if not args.no_upload and not args.audio_only:
        try:
            upload_to_gdrive(str(out_path), folder_id=args.gdrive_folder)
        except Exception as e:
            print(f"Warning: Google Drive upload failed — {e}")

    # 4. Podcast mode — LLM script + two-voice TTS
    if args.podcast:
        print("Step 4/4  Generating podcast version...")
        podcast_script = generate_podcast_script(clean_text, model_name=args.podcast_model, use_hf=args.use_hf)

        print("\n--- Generated podcast script (preview) ---")
        for line in podcast_script.splitlines()[:10]:
            print(" ", line)
        print("  ...")
        print("-------------------------------------------\n")

        print("  Synthesising podcast audio (two voices)...")
        synthesize_podcast(
            podcast_script,
            str(podcast_out),
            host_voice=args.host_voice,
            guest_voice=args.guest_voice,
            speed=args.speed,
        )
        print(f"\n  Podcast (Q&A)         →  {podcast_out}")

        if not args.no_upload:
            try:
                upload_to_gdrive(str(podcast_out), folder_id=args.gdrive_folder)
            except Exception as e:
                print(f"Warning: Google Drive upload failed for podcast — {e}")

    print(f"\nDone!\n")

    # Clean up temp downloaded PDF if applicable
    if _tmp_pdf and os.path.exists(_tmp_pdf.name):
        os.unlink(_tmp_pdf.name)


if __name__ == "__main__":
    main()
