#!/usr/bin/env bash
# ==============================================================================
# LectureLens — Batch Audiobook Generation Script
# ==============================================================================

# Use conda environment python if available, else system python
if [ -f "/home/atharva/miniconda3/envs/pdf2speech/bin/python" ]; then
    PYTHON_CMD="/home/atharva/miniconda3/envs/pdf2speech/bin/python"
else
    PYTHON_CMD="python"
fi

PDF_INPUT="${1:-textbook.pdf}"
PAGES="${2:-1-50}"
OFFSET="${3:-0}"
OUT_DIR="${4:-./Audiobook/Output}"

$PYTHON_CMD lecturelens.py "$PDF_INPUT" \
    --per-page \
    --pages "$PAGES" \
    --offset "$OFFSET" \
    --model "qwen3.8:latest" \
    --out-dir "$OUT_DIR"
