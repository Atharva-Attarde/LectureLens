#!/usr/bin/env bash
# ==============================================================================
# Overnight Batch Audiobook Generation for Oppenheim
# Total: 208 Pages across 6 Selected Chapters/Sections
# Vision Model: qwen3.8:latest
# ==============================================================================

# Use conda environment python if available, else system python
if [ -f "/home/atharva/miniconda3/envs/pdf2speech/bin/python" ]; then
    PYTHON_CMD="/home/atharva/miniconda3/envs/pdf2speech/bin/python"
else
    PYTHON_CMD="python"
fi

$PYTHON_CMD pdf2speech.py Oppenheim.pdf \
    --per-page \
    --pages "10-61,99-135,153-166,206-224,624-683,718-743" \
    --offset 23 \
    --model "qwen3.8:latest" \
    --out-dir ./Audiobook/Open
