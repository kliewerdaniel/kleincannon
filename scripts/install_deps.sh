#!/usr/bin/env bash
# kleincannon dependency installer.
# Run from the repo root:  bash scripts/install_deps.sh
set -euo pipefail

echo "== kleincannon install =="

# 1. Python venv + python deps
echo "[1/4] python venv + requirements"
python3 -m venv venv
./venv/bin/pip install -U pip
./venv/bin/pip install -r requirements.txt

# 2. ffmpeg (Homebrew) — required for assembly. libass is NOT available on the
#    bottled formula; kleincannon rasterizes captions with Pillow + overlay.
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[2/4] installing ffmpeg (brew)"
  brew install ffmpeg || echo "  brew not found — install ffmpeg manually"
else
  echo "[2/4] ffmpeg present: $(command -v ffmpeg)"
fi

# 3. ComfyUI (the image backend) — cloned next to this repo by default.
#    Override with KLEIN_COMFY_DIR env var if you keep it elsewhere.
COMFY_DIR="${KLEIN_COMFY_DIR:-$HOME/Documents/Projects/image/ComfyUI}"
if [ ! -d "$COMFY_DIR" ]; then
  echo "[3/4] ComfyUI not found at $COMFY_DIR"
  echo "      clone it: git clone https://github.com/comfyanonymous/ComfyUI $COMFY_DIR"
  echo "      then: cd $COMFY_DIR && pip install -r requirements.txt"
else
  echo "[3/4] ComfyUI present at $COMFY_DIR"
fi

# 4. speech-swift (primary F5-TTS engine, native Apple-Silicon MLX) — a Swift
#    package, NOT pip. If unavailable, tts.py falls back to the pip `f5-tts`
#    package (already installed in step 1).
if command -v speech >/dev/null 2>&1; then
  echo "[4/4] speech (speech-swift) present"
else
  echo "[4/4] speech-swift not on PATH — primary TTS engine unavailable."
  echo "      install it (macOS, needs Xcode command line tools):"
  echo "        git clone https://github.com/soniqo/speech-swift"
  echo "        cd speech-swift && make install"
  echo "      meanwhile tts.py will use the pip 'f5-tts' fallback."
fi

echo
echo "== done =="
echo "Next:  python scripts/fetch_models.py   # download F5 + FLUX.2-klein weights"
echo "Then:  ./kc doctor                      # verify the setup"
echo "       ./kc all --topic '...' --manual-script \$'line1\nline2'"
