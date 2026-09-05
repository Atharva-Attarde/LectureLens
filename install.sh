#!/usr/bin/env bash
# Install Pdf2Speech dependencies
# Usage: bash install.sh [optional: path/to/pip]

PIP="${1:-pip}"

echo "Installing Pdf2Speech dependencies..."
"$PIP" install -r requirements.txt

echo ""
echo "Checking for ffmpeg..."
if ! command -v ffmpeg &>/dev/null; then
    echo "  ffmpeg not found in PATH."
    echo "  Install it with one of:"
    echo "    conda install -c conda-forge ffmpeg"
    echo "    brew install ffmpeg          # macOS"
    echo "    sudo apt install ffmpeg      # Ubuntu/Debian"
else
    echo "  ffmpeg found: $(command -v ffmpeg)"
fi

echo ""
echo "Done! Test with:"
echo "  python pdf2speech.py --list-voices"
