# kleincannon

A **self-contained, general-purpose vertical-video generator** for TikTok / Reels /
Shorts. Give it a topic (or paste a script), a reference voice, and a visual style —
it writes a spoken monologue, voices it, aligns the words, renders one cinematic
image per beat, burns in karaoke captions, and cuts a 1080×1920 video.

No cloud APIs. No brand- or product-specific logic. Any topic, any purpose.

> **Not for commercial use (TTS license).** The voice engine is F5-TTS
> (`CC-BY-NC-4.0`, non-commercial). The image model is FLUX.2-klein (Apache-2.0).
> The user intends kleincannon for personal/non-commercial work; if you ever
> commercialize, swap `tts.py` for a permissive engine (e.g. Kokoro-82M).

---

## What's inside

| Stage | Does | Engine |
|-------|------|--------|
| `script`   | monologue + beats (pasted **or** LLM-written) | gemma4 *(optional)* |
| `tts`      | voice the script, server-less | **F5-TTS-MLX** (`speech` CLI) |
| `align`    | word-level timestamps, mapped to beats | faster-whisper |
| `prompts`  | one image prompt per beat (+ local fallback) | gemma4 *(optional)* |
| `images`   | render the beat visuals | **FLUX.2-klein-4B** via ComfyUI |
| `captions` | karaoke caption PNGs (no libass here) | Pillow |
| `assemble` | Ken Burns + concat + caption overlay + audio | ffmpeg |

Two long-lived services at most: the **LLM** (optional) and **ComfyUI** (auto-
launched by `images`). TTS is a one-shot subprocess — no server.

---

## Install

```bash
bash scripts/install_deps.sh        # venv, python deps, ffmpeg, ComfyUI check
python scripts/fetch_models.py      # download F5 + FLUX.2-klein weights
./kc doctor                         # verify everything is wired up
```

`install_deps.sh` uses Homebrew ffmpeg. Note: the bottled ffmpeg on macOS ships
**without libass/drawtext**, so captions are rasterized with Pillow to transparent
PNGs and composited with ffmpeg's `overlay` filter — no change needed from you.

The `speech` CLI (speech-swift, the native MLX F5 engine) is a Swift package; if
it isn't installed, `tts.py` falls back to the pip `f5-tts` package automatically.

---

## CLI quick start

```bash
# one-shot: topic -> final mp4 (pasted script)
./kc all --topic "why a single typo kills a real-estate deal" \
         --manual-script $'A single transposed number just lost you the deal.\nYou typed a nine where the eight should be.\nThat is a quarter-million swing in the cap rate.'

# or stage by stage
./kc new     --topic "..." --manual-script $'beat one\nbeat two'   # create episode
./kc tts     -e <id> --voice chris
./kc align   -e <id>
./kc prompts -e <id>
./kc images  -e <id> [--fast]        # --fast = low-res quick render
./kc captions -e <id>
./kc build   -e <id>
./kc sheet   -e <id>                 # show the episode manifest
```

Omit `-e` to target the most recently modified episode. Use `--fast` on `images`
to render tiny low-step frames for a quick end-to-end test (assembly adapts to the
real rendered size).

---

## Web UI

```bash
./venv/bin/python web/server.py      # http://127.0.0.1:8400
```

The control-room UI exposes every knob: topic/purpose, manual-or-AI script, voice,
visual style, caption colours, Ken Burns zoom, motion direction, and an optional
free-text CTA (off by default, never auto-burned). Progress streams live over SSE.

---

## Customization

All defaults live in `kleincannon/config.py` and can be overridden per run
(web form or `config.push_overrides`).

| Dimension | Set via | Default |
|---|---|---|
| Topic / purpose | `topic` + `purpose` | manual |
| Script | pasted `manual_script` **or** gemma4 | — |
| Voice | any clip in `voices/<name>` (+ `<name>.txt`) | `chris` |
| Visual style | `style_suffix` + per-beat prompts | moody cinematic |
| Caption styling | accent/white colours, font fracs | cyan-on-white lower third |
| Motion | per-beat Ken Burns `in/out/left/right` | `in` |
| Output size | `WIDTH/HEIGHT/FPS`, FAST vs native | 1080×1920 @ 30 |
| Optional CTA | `cta` free text (off, not burned) | empty |

### Reference voices

F5-TTS is **zero-shot**: it needs a short clip + its exact transcript. Drop
`voices/<name>.wav` and `voices/<name>.txt` (see `voices/README.md`). The default
`chris` clip ships in-repo; author `voices/chris.txt` before first TTS run.

---

## Caveats

- **ffmpeg has no libass** on this Mac → captions use Pillow PNGs + `overlay`.
- **LLM is optional** — manual-script mode (and the prompts-stage local fallback)
  work fully offline. A full AI run needs `llama-server` serving gemma4.
- **ComfyUI auto-launches** for `images`. The repo's `klein.json` workflow has its
  LoRA Loader bypassed by default (see `comfy.py`); point `UnetLoaderGGUF` at your
  preferred `flux-2-klein-4b-*.gguf` in the workflow if you want a different quant.
- Image generation is GPU-bound and **slow** (~minutes per beat at native res). Use
  `--fast` to validate the whole pipeline before a real render.
