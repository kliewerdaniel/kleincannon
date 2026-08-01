# kleincannon — Design Plan (v1, for review)

> **Status:** PLANNING. No pipeline code has been written yet. This document is the
> first deliverable and is awaiting your review before implementation begins.
>
> **Goal in one line:** A single, self-contained repository that turns a topic (or a
> pasted script) into a finished, captioned, vertical (9:16) short — using
> **F5TTS-MLX** for voice, **FLUX.2-klein (GGUF)** for visuals, and **gemma4
> (llama.cpp)** for text, with **no music stage** and **as few running services as
> possible**.

---

## 0. TL;DR for review

| Capability | Engine (new) | How it runs | Persistent service? |
|---|---|---|---|
| Script + image prompts | **gemma4** GGUF via llama.cpp | `llama-server` (optional) | Only if you want AI text. **Manual mode skips it.** |
| **Voice (TTS)** | **F5TTS-v1-Base-MLX-fp16** | `speech` **CLI** subprocess (no server) | **None ✓** |
| **Images** | **FLUX.2-klein-4B GGUF** | **ComfyUI** klein workflow | **ComfyUI** — auto-launched by the pipeline |
| Word alignment | faster-whisper | in-process | none |
| Captions (karaoke) | Pillow + ffmpeg `overlay` | in-process | none |
| Assembly | ffmpeg | in-process | none |
| **Music** | — | **removed** | none |

**Net change vs tokcannon:** 4 external services (llama, vox, ComfyUI, ACE-Step)
→ **2** (llama *optional*, ComfyUI *auto-launched*). TTS is now server-less.
Music is gone entirely.

> **Scope:** kleincannon is a **general-purpose** vertical-video generator — make
> videos about *any* topic, for *any* purpose (educational, personal, creative,
> promotional, whatever). It is **not** hard-wired to any one product or brand.
> Topic, script, voice, visual style, caption styling, and an optional call-to-action
> are all fully configurable per episode. See §1.3.
>
> **TTS license — RESOLVED.** F5TTS-v1-Base-MLX-fp16 is `CC-BY-NC-4.0`
> (non-commercial). For **personal / non-commercial** use that is fine. If you ever
> commercialize, you'd swap the TTS engine behind the same `tts.py` interface (§9.1).
> No blocker for the current intent. Please decide before we build the TTS stage for real.

---

## 1. Goals & non-goals

### Goals
1. **One repo, nothing missing.** Every script, workflow, voice sample, config,
   and the *code to fetch the two model weights* lives in the repo. No scattered
   `~/Documents/Projects/*` dependencies.
2. **General-purpose & highly customizable.** kleincannon is *not* tied to one
   product or brand. Any topic, any purpose. See §1.3.
3. **Minimize services.** Prefer in-process or one-shot CLI invocations over
   long-lived servers. The only required server is ComfyUI, and the pipeline
   starts it itself.
4. **Swap the two heavy engines.**
   - TTS: `vox`/Qwen3-TTS → **F5TTS-MLX** (via `speech-swift` CLI).
   - Images: ComfyUI **Qwen-Image** workflow → ComfyUI **FLUX.2-klein GGUF** klein workflow.
5. **Drop music.** No ACE-Step, no `music.py`, no `MUSIC_DB` mixing.
6. **Keep what works.** gemma4 text gen (optional), faster-whisper align,
   Pillow/ffmpeg karaoke captions, the FastAPI web UI, the `tok`-style CLI.
7. **Self-contained captions.** ffmpeg on this Mac lacks `libass`, so we keep the
   Pillow→transparent-PNG→`overlay` approach (no change needed).

### Non-goals (this iteration)
- No cloud/API calls. Everything local (matches your sovereignty preference).
- No new video model (still stills + Ken Burns + captions).
- No multi-tenant / concurrent runs (single run at a time, as today).
- Not re-litigating the no-text-in-images rule (kept, hardened).
- Not building any product-specific logic (no DealCannon-only behavior).

#### 1.3 Customizability surface (first-class)

kleincannon should feel like a general tool, not a single-brand machine. Every
run is configured through the episode manifest + CLI/UI params, with sensible
defaults but no required product coupling:

| Dimension | How it's set | Default |
|---|---|---|
| **Topic / purpose** | `topic` + optional `purpose`/`hook` fields | none required in manual mode |
| **Script** | AI-generated (gemma4, optional) **or** pasted `manual_script` | — |
| **Voice** | any reference clip in `voices/` (`name.wav` + `name.txt`) | `chris` |
| **Visual style** | `style_suffix` + per-beat prompts (LLM or local fallback) | moody cinematic (editable) |
| **Caption styling** | accent/white colors, font fracs, placement | cyan-on-white, lower third |
| **Motion** | per-beat Ken Burns direction (`in/out/left/right`) | `in` |
| **Output size / quality** | `WIDTH/HEIGHT/FPS`, `FAST` vs native profile | 1080×1920 @ 30 |
| **Optional CTA / URL** | `cta` field (free text, shown nowhere automatically) | empty (off) |
| **Music** | **removed** — no bed | n/a |

The `cta` field is just a string the user can drop into their script/prompt if
they want; it is **not** auto-burned or brand-locked. This keeps the tool usable
for any subject.

---

## 2. Model inventory & licenses

| Model | Source | License | Commercial use? | Notes |
|---|---|---|---|---|
| **F5TTS-v1-Base-MLX-fp16** | `aufklarer/F5TTS-v1-Base-MLX-fp16` | **CC-BY-NC-4.0** | **No** ⚠️ | 336M, 24 kHz, native Swift/MLX, zero-shot voice cloning (needs ref clip + transcript). 670 MB. |
| **FLUX.2-klein-4B-GGUF** | `unsloth/FLUX.2-klein-4B-GGUF` | **Apache-2.0** | **Yes** ✓ | 4B rectified-flow, GGUF (Q2–Q8 + BF16/F16). Loaded via ComfyUI-GGUF (city96). ~13 GB VRAM at Q8, less quantized. |
| **gemma4** (text) | your existing GGUF | (your pick) | check yours | llama.cpp server; optional. Manual mode bypasses it. |

**Self-contained fetch:** a `scripts/fetch_models.py` downloads both HF repos into
`models/` on first setup (weights are git-ignored; the *fetch code* is in the repo,
so "everything" is present in source).

---

## 3. Architecture

```
topic ──▶ script ──▶ tts(F5) ──▶ align(whisper) ──▶ prompts(gemma, opt) ──▶ images(klein/ComfyUI) ──▶ captions(Pillow) ──▶ assemble(ffmpeg) ──▶ mp4
              │                                                                                               
        manual_script ─┘ (paste your own monologue → skips gemma entirely)                                     
```

- **`script`** writes/accepts the monologue. AI mode calls gemma4 (optional);
  manual mode (`from_text`) skips the LLM.
- **`tts`** shells out to `speech speak …` (F5TTS-MLX) → `audio/voice.wav`.
  No server. Reads a reference clip + its transcript from `voices/`.
- **`align`** transcribes `voice.wav` with faster-whisper → per-beat `audio_start/end`.
- **`prompts`** writes one image prompt per beat (gemma4, with local fallback).
- **`images`** renders one still per beat via **ComfyUI klein workflow**
  (auto-launched if not already up).
- **`captions`** rasterizes karaoke overlays with Pillow → `captions/*.png` + `words.json`.
- **`assemble`** Ken Burns + overlay captions + mix voice → `episode.mp4`.
  **No music branch.**

---

## 4. Proposed repo layout

```
kleincannon/
├── PLAN.md                      # this document
├── README.md                    # short pointer + status
├── requirements.txt             # fastapi uvicorn faster-whisper Pillow numpy
├── .gitignore                   # venv/, models/, episodes/* media, __pycache__
├── scripts/
│   ├── fetch_models.py          # download F5TTS + FLUX.2-klein GGUF into models/
│   └── install_deps.sh          # brew ffmpeg; install speech-swift; (ComfyUI hint)
├── kleincannon/
│   ├── __init__.py
│   ├── config.py                # paths, endpoints, knobs (rewritten)
│   ├── llm.py                   # gemma4 client (kept, optional)
│   ├── episode.py               # Episode/Beat manifest (kept)
│   ├── tts.py                   # NEW: F5TTS-MLX via speech CLI
│   ├── align.py                 # kept (faster-whisper)
│   ├── prompts.py               # kept (gemma4 + local fallback)
│   ├── comfy.py                 # REWRITE: klein workflow + ensure_server()
│   ├── images.py                # REWRITE: klein GGUF via comfy
│   ├── captions.py              # kept
│   ├── assemble.py              # kept, music branch removed
│   └── workflows/
│       └── klein.json           # the ComfyUI klein workflow (committed to repo)
├── voices/                      # reference clips + transcripts for F5 cloning
│   ├── chris.wav
│   └── chris.txt                # exact transcript of chris.wav
├── web/                         # FastAPI server + vanilla-JS UI (music params dropped)
│   ├── server.py
│   └── static/{index.html,app.js,styles.css}
└── episodes/                    # one folder per generated video (+ .gitkeep)
```

**Why `workflows/klein.json` is committed:** it makes image generation
reproducible from the repo alone — no external file dependency. We copy your
existing ComfyUI klein workflow into the repo at build time.

**Why `voices/` is committed:** F5TTS needs a reference clip **and its transcript**
to clone a voice. Bundling both keeps the repo complete.

---

## 5. Stage-by-stage changes from tokcannon → kleincannon

| Stage | tokcannon | kleincannon | Action |
|---|---|---|---|
| `script` | gemma (optional) | gemma4 (optional) | keep |
| `tts` | vox (Qwen3-TTS, server on :7860) | **F5TTS-MLX `speech` CLI** | **rewrite** |
| `align` | faster-whisper | faster-whisper | keep |
| `prompts` | gemma (optional, local fallback) | gemma4 (optional, local fallback) | keep |
| `images` | ComfyUI Qwen-Image | ComfyUI **klein GGUF** | **rewrite** (workflow + auto-launch) |
| `music` | ACE-Step bed | — | **delete** |
| `captions` | Pillow overlays | Pillow overlays | keep |
| `assemble` | voice+music mix | voice only | **trim music branch** |
| `web` | server + UI | server + UI (no music) | trim params |
| `config` | vox/comfy/ace/llm paths | f5/comfy/gemma paths | **rewrite** |

---

## 6. TTS deep-dive — F5TTS-MLX via `speech-swift`

**What it is:** a native Apple-Silicon MLX bundle. Consumed by the open-source
[`speech-swift`](https://github.com/soniqo/speech-swift) library (module `F5TTS`)
and its `speech` CLI. **No Python package, no server.** We invoke it as a subprocess.

**Invocation (from README):**
```bash
speech speak "Text to synthesize." --engine f5 \
  --voice-sample voices/chris.wav \
  --f5-reference-text "Transcript of the reference clip." \
  -o episodes/<id>/audio/voice.wav
```
The model id `aufklarer/F5TTS-v1-Base-MLX-fp16` can be a **local path** once
downloaded by `fetch_models.py` (keeps it offline/self-contained).

**Reference sample + transcript:** stored in `voices/<name>.wav` + `voices/<name>.txt`.
`tts.py` reads both; the transcript is required for cloning.

**Long-script handling:** F5-TTS is happiest with reference ≤10 s and synthesizes
in chunks. We will split the monologue into sentences/clauses, call `speech speak`
per chunk, and concatenate the WAVs with ffmpeg into one `voice.wav` (mirrors what
`vox`'s `generate_long` did). Implementation detail, not a design risk.

**Output:** 24 kHz / mono. `assemble` already resamples through ffmpeg, so no
extra step — but note voice is **24 kHz** (telephone-ish). Acceptable for shorts;
flagged in §9.

**Env safety:** launched with `env -u PYTHONPATH -u PYTHONHOME` (matches the
hardened pattern we already use for ComfyUI/vox — the agent venv leak must never
reach a py3.14 process).

---

## 7. Images deep-dive — FLUX.2-klein GGUF via ComfyUI

**The only remaining service is ComfyUI.** To honor "you shouldn't have to run
ComfyUI yourself," `comfy.py` gains an `ensure_server()` that **starts ComfyUI
from the repo's bundled interpreter** if it isn't already up (same pattern as the
old `tts.ensure_server` for vox). So `kc images -e <id>` just works.

**Workflow is in-repo:** `kleincannon/workflows/klein.json` (copied from your
ComfyUI klein workflow). `images.py` loads it, patches positive/negative prompt,
seed, and latent size, posts to ComfyUI, polls `/history`, copies the result out.
No change to the *mechanism* — only the workflow file and size defaults.

**Size/steps:** FLUX.2-klein is fast (sub-second native, but ComfyUI GGUF + our
hardware will be slower). We keep a `FAST` profile (lower steps / smaller latent)
for pipeline validation and a native profile for final. Exact numbers set from the
klein workflow's native KSampler when we copy it in.

**ComfyUI itself:** not bundled in git (large). `install_deps.sh` documents the
one-time `git clone` + `pip install` (it already exists at
`~/Documents/Projects/image/ComfyUI` on this machine; the repo just needs to know
its path in `config.py`). ComfyUI-GGUF (city96) node must be installed for the
klein GGUF to load.

---

## 8. "No missing services" strategy

| Need | How it's satisfied by the repo |
|---|---|
| Text model | Optional; manual mode removes the dependency entirely |
| Voice | `speech` CLI — no server, invoked per run |
| Images | ComfyUI **auto-launched** by `comfy.ensure_server()` |
| Align / captions / assemble | pure-Python + ffmpeg (system `ffmpeg` via Homebrew) |
| Model weights | `scripts/fetch_models.py` downloads both into `models/` |
| Voice sample | committed in `voices/` |
| ComfyUI workflow | committed in `workflows/klein.json` |

**Still external (acceptable, documented):**
- **Homebrew `ffmpeg`** (system). Could be bundled later; not worth it now.
- **ComfyUI install + ComfyUI-GGUF node** (one-time, documented in `install_deps.sh`).
- **The two model weights** (downloaded once by the repo's own script).

This is the practical floor for "self-contained": code + config + workflow + voice
are all in git; weights + ComfyUI are fetched/installed by repo-provided scripts.

---

## 9. Risks & open decisions

### 9.1 TTS license (non-commercial use — RESOLVED)
F5TTS-v1-Base-MLX-fp16 is `CC-BY-NC-4.0` (non-commercial). For **personal /
non-commercial** video creation that is fine, so this is no longer a blocker.
The `tts.py` interface is still designed behind one function so that, *if* you ever
commercialize videos made with kleincannon, you can swap in a permissively-licensed
local TTS (e.g. Kokoro-82M, Apache-2.0) without touching the rest of the pipeline.

### 9.2 F5TTS install path not yet verified
`speech-swift` is a Swift package. Need to confirm install method
(`brew` / `swift build` / prebuilt) on this Mac and that the `speech` binary lands
on `PATH`. `install_deps.sh` will codify it; flagged until verified.

### 9.3 24 kHz voice quality
F5TTS-MLX outputs 24 kHz. Fine for shorts; if it sounds thin we can upsample or
revisit the model. Not a blocker.

### 9.4 ComfyUI is still a service
You accepted this ("if we need to start the server that will be fine too"). The
auto-launch keeps it hands-off. If you later want **zero** services, the alternative
is running the klein GGUF through a non-ComfyUI loader (e.g. diffusers+GGUF, or a
llama.cpp image path) — out of scope for v1, noted as a future option.

### 9.5 gemma4 text is optional
Kept as today: manual-script mode bypasses it, and `prompts.py` has a deterministic
local fallback. The LLM is the only *other* server, and it's avoidable.

---

## 10. Config reference (proposed new keys in `config.py`)

| Key | Default (proposed) | Meaning |
|---|---|---|
| `F5_MODEL_DIR` | `models/F5TTS-v1-Base-MLX-fp16` | local F5TTS bundle path |
| `F5_BIN` | `speech` | the speech-swift CLI binary |
| `VOICES_DIR` | `voices/` | reference clips + transcripts |
| `DEFAULT_VOICE` | `chris` | name (resolves `chris.wav`+`chris.txt`) |
| `COMFY_DIR` | (your ComfyUI path) | ComfyUI install |
| `COMFY_URL` | `http://127.0.0.1:8188` | (auto-launched) |
| `COMFY_WORKFLOW` | `workflows/klein.json` | in-repo klein workflow |
| `GEN_WIDTH/HEIGHT` | from klein workflow | native latent |
| `FAST_GEN_WIDTH/HEIGHT`, `FAST_STEPS` | test profile | pipeline validation |
| `LLM_GGUF`, `LLM_URL` | gemma4 (kept) | optional text gen |
| `WIDTH/HEIGHT/FPS` | 1080/1920/30 | output |
| `ZOOM_MAX`, `CRF`, `TAIL_SECONDS` | kept | assembly |
| caption knobs | kept | karaoke styling |
| `ALIGN_MODEL` | `small.en` | faster-whisper |

(`MUSIC_DB`, `VOX_*`, `ACE_*` removed.)

---

## 11. Install / setup (one pass)

```bash
# 1. clone / create kleincannon, make venv
cd ~/Documents/Projects/kleincannon
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # fastapi uvicorn faster-whisper Pillow numpy

# 2. system deps
brew install ffmpeg
./scripts/install_deps.sh                # installs speech-swift; documents ComfyUI + GGUF node

# 3. model weights (repo-provided downloader)
python scripts/fetch_models.py           # -> models/F5TTS-v1-Base-MLX-fp16, models/FLUX.2-klein-4B-*.gguf

# 4. voice sample already in voices/ (chris.wav + chris.txt)
```

---

## 12. Usage

```bash
# CLI
./kc new "topic" -b 6                       # write monologue (or use manual mode)
./kc tts -e <id> --voice chris
./kc align -e <id>
./kc prompts -e <id>
./kc images -e <id> --fast                  # ComfyUI auto-launches if needed
./kc captions -e <id>
./kc build -e <id>                          # assemble mp4 (no music)

# Web UI
source venv/bin/activate
env -u PYTHONPATH -u PYTHONHOME venv/bin/python web/server.py   # http://127.0.0.1:8400
```

(Manual-script mode in the UI pastes the monologue → skips gemma4.)

---

## 13. Implementation roadmap (phases, for after your review)

1. **Scaffold repo** — `git init`, folders, `requirements.txt`, `config.py`, `episode.py`, `llm.py` (ported).
2. **TTS stage** — `tts.py` invoking `speech` CLI; verify a 6-beat voice renders (resolve §9.1 license decision first).
3. **Images stage** — copy `klein.json` into repo; rewrite `comfy.py` (+`ensure_server`) and `images.py`; verify one beat renders.
4. **Trim music** — delete `music.py`; remove music branch from `assemble.py` + web params.
5. **Captions/assemble** — port as-is (Pillow overlays, voice-only mix).
6. **Web UI** — drop music controls; point at new stages.
7. **fetch_models.py + install_deps.sh** — self-containment scripts.
8. **End-to-end verify** — full run (manual script) → playable 1080×1920 mp4 with captions.
9. **README + doc** — final docs.

---

## 14. Verification plan

- `kc doctor` reports: ffmpeg present, `speech` on PATH, F5 model dir present,
  ComfyUI reachable-or-auto-launchable, gemma4 optional.
- A manual-script run produces `episodes/<id>/<id>.mp4` at 1080×1920, correct
  duration, captions visible, **no music track**.
- Idempotent re-runs; `--fast` validates the pipeline in minutes.
