#!/usr/bin/env bash
# ==============================================================================
# LectureLens — Batch Audiobook Generation Script
# ==============================================================================

if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
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
