"""Machine-local paths and service endpoints. Everything else imports from here.

kleincannon is a self-contained, general-purpose vertical-video generator.
All tunable knobs live here and can be overridden at runtime (web UI / CLI).
"""
from __future__ import annotations

import sys
from pathlib import Path

HOME = Path.home()
PROJECTS = HOME / "Documents" / "Projects"

ROOT = PROJECTS / "kleincannon"
EPISODES = ROOT / "episodes"
WORKFLOWS = ROOT / "kleincannon" / "workflows"
MODELS = ROOT / "models"
VOICES = ROOT / "voices"
PROMPTS = ROOT / "prompts"

# --- LLM (llama.cpp, gemma4) — OPTIONAL. Manual-script mode bypasses it. ---
LLAMA_SERVER_BIN = Path("/usr/local/bin/llama-server")
LLM_GGUF = PROJECTS / "chopnscrwbot" / "gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf"
LLM_HOST = "127.0.0.1"
LLM_PORT = 8080
LLM_URL = f"http://{LLM_HOST}:{LLM_PORT}"
LLM_CTX = 8192
LLM_NGL = 99

# --- TTS (F5TTS-MLX, via speech-swift CLI) — server-less, invoked as a subprocess ---
# `speech` is the speech-swift CLI. If it is not on PATH, tts.py falls back to the
# pip-installable `f5-tts` package (Apache-2.0). See scripts/install_deps.sh.
F5_BIN = "speech"
F5_MODEL_DIR = MODELS / "F5TTS-v1-Base-MLX-fp16"
F5_FALLBACK_PKG = "f5_tts"          # module used by the python fallback
DEFAULT_VOICE = "chris"            # resolves voices/chris.wav + voices/chris.txt

# --- Images (ComfyUI + FLUX.2-klein GGUF) — ComfyUI is auto-launched if needed ---
COMFY_DIR = PROJECTS / "image" / "ComfyUI"
COMFY_URL = "http://127.0.0.1:8188"
COMFY_OUTPUT = COMFY_DIR / "output"
COMFY_WORKFLOW = WORKFLOWS / "klein.json"   # committed in-repo (FLUX.2-klein)
# ComfyUI is expected to be RUN BY YOU on http://127.0.0.1:8188 before `kc images`.
# klein poisons the Apple-Silicon MPS pool when the agent relaunches the server
# between beats, so we do NOT auto-launch or kill it — we just connect. Flip
# COMFY_AUTO_LAUNCH to True if you'd rather the agent launch it (with --lowvram).
COMFY_AUTO_LAUNCH = False
COMFY_PYTHON = COMFY_DIR / "main.py"
COMFY_LAUNCH_ARGS: list[str] = ["--lowvram"]   # used only when COMFY_AUTO_LAUNCH=True
# ComfyUI runs under its own venv (has comfy_aimdo / alembic / torch). Use that
# interpreter rather than the system python, which lacks ComfyUI's deps.
COMFY_VENV_PYTHON = PROJECTS / "image" / "venv" / "bin" / "python3"

# --- Video ---
FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
WIDTH, HEIGHT, FPS = 1080, 1920, 30

# ComfyUI's FLUX.2-klein latent is typically 1024x1024; we crop to 1080 at
# assembly. Keep these in sync with the klein workflow's native EmptyLatent size
# (comfy.workflow_settings() reads the real value from the file for `kc doctor`).
# Native render size of the working klein graph (vertical 9:16, as you exported it).
GEN_WIDTH, GEN_HEIGHT = 1080, 1920

# Fast/test render profile — low-res + fewer steps so a full end-to-end run
# finishes in minutes instead of per-frame latency. `kc images --fast` uses it;
# assembly reads the real rendered size so it cuts together correctly.
FAST_GEN_WIDTH, FAST_GEN_HEIGHT = 540, 960   # 9:16 low-res test profile
FAST_STEPS = 4

TAIL_SECONDS = 1.2      # silence after last word before video ends

# --- Caption styling (karaoke burn-in). Overridable from web UI / CLI. ---
CAPTION_FONT_MAX_FRAC = 0.075   # largest caption font as a fraction of frame height
CAPTION_FONT_MIN_FRAC = 0.030   # floor so very long lines stay readable
CAPTION_WIDTH_BUDGET = 0.86     # caption block may use this fraction of frame width
CAPTION_BLOCK_MAX_FRAC = 0.34    # caption block may use this fraction of frame height
CAPTION_TOP_FRAC = 0.80         # vertical center of the caption block (lower third)
CAPTION_ACCENT = (0, 229, 255, 255)    # spoken-word colour (cyan)
CAPTION_WHITE = (255, 255, 255, 255)   # upcoming-word colour (white)
CAPTION_BACKING = (16, 16, 16, 178)     # semi-transparent plate behind the text
CAPTION_STROKE = (0, 0, 0, 255)         # text outline

# --- Ken Burns / assembly ---
ZOOM_MAX = 1.14        # max zoom on a held still (more looks jittery)
CRF = 19               # H.264 quality (lower = better, slower)

# --- Alignment ---
ALIGN_MODEL = "small.en"   # faster-whisper model size for word timestamps

CTA = ""               # optional free-text call-to-action; never auto-burned


# ----------------------------------------------------------------------------
# Runtime override support. The web server applies a run's parameters by
# mutating these module-level attributes for the duration of one pipeline run,
# then restores the originals. Runs are sequential so this is safe.
# ----------------------------------------------------------------------------
_OVERRIDE_STACK: list[dict] = []


def push_overrides(overrides: dict) -> None:
    """Set the given config attributes, remembering originals for restore()."""
    saved = {}
    for key, value in overrides.items():
        if hasattr(config, key):
            saved[key] = getattr(config, key)
            setattr(config, key, value)
    _OVERRIDE_STACK.append(saved)


def pop_overrides() -> None:
    """Restore the most recently pushed set of overrides."""
    if not _OVERRIDE_STACK:
        return
    saved = _OVERRIDE_STACK.pop()
    for key, value in saved.items():
        setattr(config, key, value)


# `config` self-reference used by push/pop (kept local to avoid import cycles).
config = sys.modules[__name__]
