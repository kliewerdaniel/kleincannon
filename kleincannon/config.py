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

# --- TTS (Qwen3-TTS via mlx-audio) — server-less, in-process on Apple Silicon ---
# Qwen3-TTS is an in-context-learning voice-cloning model. tts.py drives it
# through the `mlx-audio` package (MLX, Apple-Silicon). It needs a reference
# clip (voices/<name>.wav) plus the whisper-transcribed transcript of that clip;
# it speaks any text in that voice. No long-lived TTS server, no manual .txt.
QWEN3_MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit"
QWEN3_REF_STT = "base"            # faster-whisper model for ref transcription
DEFAULT_VOICE = "chris"            # resolves voices/chris.wav (full clip)

# --- TTS pacing (human-in-the-loop parameter) ---
# The cloned voice reads slightly slow by default; the user asked for a touch
# of speed-up so the narration keeps pace. 1.0 = natural, >1.0 = faster.
DEFAULT_TTS_SPEED = 1.12

# --- Visual "style" library (human-in-the-loop parameter) ---
# Every generated video gets one of these style pre-prompts appended to each
# beat's image prompt. The default FLUX.2-klein output drifts blue/teal and all
# clips look alike, so each episode now draws a DISTINCT style (deterministically
# from its id, or explicitly via --style / the web form) so videos don't repeat
# looks. Each suffix keeps the "absolutely no text …" guard. The style NAME is
# stored on the episode manifest and fed to the learning engine so the bandit can
# learn which looks perform — i.e. style is both adjustable by you AND observable
# by the optimizer.
STYLE_CATALOG = [
    {"name": "Moody Cinematic", "palette": "teal & amber",
     "suffix": "cinematic editorial photograph, vertical 9:16 composition, shot on 35mm, "
               "shallow depth of field, moody directional window light, desaturated teal and amber "
               "palette, film grain, high detail, absolutely no text, no writing, no signage, "
               "no readable screens, no numbers, no logos"},
    {"name": "Warm Documentary", "palette": "golden amber",
     "suffix": "warm natural documentary photograph, vertical 9:16, available light, golden hour, "
               "35mm film look, gentle film grain, earthy amber and cream palette, high detail, "
               "absolutely no text, no writing, no signage, no readable screens, no numbers, no logos"},
    {"name": "Noir", "palette": "monochrome",
     "suffix": "high-contrast black and white film noir photograph, vertical 9:16, hard directional "
               "lighting, deep shadows, stark monochrome, dramatic chiaroscuro, film grain, "
               "absolutely no text, no writing, no signage, no readable screens, no numbers, no logos"},
    {"name": "Vivid Editorial", "palette": "saturated",
     "suffix": "vibrant fashion-editorial photograph, vertical 9:16, bold saturated color, glossy "
               "studio light, punchy contrast, sharp detail, absolutely no text, no writing, no "
               "signage, no readable screens, no numbers, no logos"},
    {"name": "Soft Pastel", "palette": "pink & sky",
     "suffix": "soft pastel-toned lifestyle photograph, vertical 9:16, airy diffuse light, gentle "
               "pinks and sky blues, dreamy low-contrast, fine grain, absolutely no text, no writing, "
               "no signage, no readable screens, no numbers, no logos"},
    {"name": "Cyberpunk Neon", "palette": "magenta & cyan",
     "suffix": "cyberpunk neon photograph, vertical 9:16, electric magenta and cyan glow, dark wet "
               "streets, volumetric light, high contrast, absolutely no text, no writing, no signage, "
               "no readable screens, no numbers, no logos"},
    {"name": "Vintage Faded", "palette": "sepia wash",
     "suffix": "faded vintage 1970s photograph, vertical 9:16, muted washed-out tones, light leaks, "
               "soft focus, warm sepia cast, absolutely no text, no writing, no signage, no readable "
               "screens, no numbers, no logos"},
]
# Override keys (runtime-set via config.push_overrides / web form). Defined here
# so push_overrides finds them as real module attributes.
PROMPT_STYLE = "auto"              # "auto" => derive deterministically per episode
IMAGE_SEED = None                  # None => deterministic per-beat seed in images.py
IMAGE_STEPS = None                 # None => workflow default (FAST_STEPS in --fast)
IMAGE_CFG = None                   # None => workflow default CFG (temperature analog)

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
# 3-word cards: keep the font well under a quarter of frame width so it reads
# as an accent, not a billboard. ~4-5% of frame height is the sweet spot for
# 1080-wide verticals (was 7.5% = 144px, which dominated the screen).
CAPTION_FONT_MAX_FRAC = 0.045   # largest caption font as a fraction of frame height
CAPTION_FONT_MIN_FRAC = 0.024   # floor so very long lines stay readable
CAPTION_WIDTH_BUDGET = 0.80     # caption block may use this fraction of frame width
CAPTION_BLOCK_MAX_FRAC = 0.30    # caption block may use this fraction of frame height
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
