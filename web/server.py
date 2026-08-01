"""kleincannon web UI — FastAPI backend + SSE live progress.

General-purpose: the run form exposes topic/purpose, script (manual or AI),
voice, visual style, caption styling, motion, output size, and an optional free-
text CTA (off by default, never auto-burned). No brand- or product-specific logic.

Pipeline runs in a worker thread; progress is broadcast over SSE. The SSE loop
is captured from inside the request handler (uvicorn owns the loop), so events
actually fire.
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from kleincannon import config
from kleincannon.episode import Episode

app = FastAPI(title="kleincannon")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_ROOT = Path(__file__).resolve().parent
_STATIC = _ROOT / "static"

# ---- SSE plumbing ----------------------------------------------------------
_subscribers: set[asyncio.Queue] = set()
_main_loop: asyncio.AbstractEventLoop | None = None


def _broadcast(event: dict) -> None:
    if _main_loop is None:
        return
    for q in list(_subscribers):
        try:
            _main_loop.call_soon_threadsafe(q.put_nowait, event)
        except Exception:
            pass


def _emit(ev_type: str, **kw) -> None:
    _broadcast({"type": ev_type, **kw})


# ---- run parameters --------------------------------------------------------
class RunParams(BaseModel):
    topic: str = ""
    purpose: str = ""
    manual_script: str = ""          # one sentence per beat, newline-separated
    use_manual: bool = True
    beats: int = 6
    voice: str = config.DEFAULT_VOICE
    style_suffix: str = ""
    caption_accent: str = "#00E5FF"
    caption_white: str = "#FFFFFF"
    zoom_max: float = config.ZOOM_MAX
    crf: int = config.CRF
    align_model: str = config.ALIGN_MODEL
    fast: bool = False
    motion: str = "in"
    cta: str = ""


def _apply_overrides(p: RunParams) -> None:
    def hx(c: str) -> tuple[int, int, int, int]:
        c = c.lstrip("#")
        r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        return (r, g, b, 255)

    config.push_overrides({
        "DEFAULT_VOICE": p.voice,
        "CAPTION_ACCENT": hx(p.caption_accent),
        "CAPTION_WHITE": hx(p.caption_white),
        "ZOOM_MAX": p.zoom_max,
        "CRF": p.crf,
        "ALIGN_MODEL": p.align_model,
        "CTA": p.cta,
    })


# ---- pipeline runner -------------------------------------------------------
def _run_pipeline(p: RunParams) -> str:
    from kleincannon.stages import script as s_stage, tts, align, prompts, images, captions, assemble

    _emit("run_start", topic=p.topic or "(manual)")

    def stage(name: str, desc: str):
        _emit("stage", name=name, desc=desc)

    try:
        stage("script", "Writing the script")
        if p.manual_script.strip():
            ep = s_stage.from_text(p.topic or "untitled", p.manual_script.strip().splitlines(),
                                   purpose=p.purpose, voice=p.voice)
        else:
            ep = s_stage.from_ai(p.topic, purpose=p.purpose, beats=p.beats, voice=p.voice)
        eid = ep.id

        stage("tts", f"Voicing with '{p.voice}' (F5TTS)")
        tts.run(eid)

        stage("align", "Aligning words to audio")
        align.run(eid)

        stage("prompts", "Composing image prompts")
        prompts.run(eid, style_suffix=p.style_suffix or prompts.STYLE_SUFFIX)

        stage("images", "Rendering visuals (FLUX.2-klein via ComfyUI)")
        images.run(eid, fast=p.fast)

        stage("captions", "Rasterizing karaoke captions")
        captions.run(eid)

        stage("assemble", "Cutting the final video")
        assemble.run(eid)

        _emit("done", result={
            "id": eid,
            "mp4": f"episodes/{eid}/{eid}.mp4",
            "duration": ep.total_duration,
        })
        return eid
    except Exception as e:  # noqa: BLE001
        _emit("error", message=str(e)[:800])
        raise
    finally:
        config.pop_overrides()


# ---- routes ----------------------------------------------------------------
@app.get("/api/health")
async def health():
    from kleincannon import comfy
    def llm_up():
        try:
            import urllib.request
            urllib.request.urlopen(config.LLM_URL + "/health", timeout=2)
            return True
        except Exception:
            return False
    return {
        "ok": True,
        "services": {"llm": llm_up(), "comfyui": comfy.is_up()},
        "busy": bool(_subscribers),
    }


@app.post("/api/run")
async def run(p: RunParams, request: Request):
    if _subscribers:
        return {"accepted": False, "error": "a run is already in progress"}
    _apply_overrides(p)
    loop = asyncio.get_event_loop()
    threading.Thread(target=_run_pipeline, args=(p,), daemon=True).start()
    return {"accepted": True}


@app.get("/api/stream")
async def stream(request: Request):
    global _main_loop
    try:
        _main_loop = asyncio.get_running_loop()
    except RuntimeError:
        pass
    q: asyncio.Queue = asyncio.Queue()
    _subscribers.add(q)
    _emit("connected")  # immediate handshake so the client knows we're live

    async def gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            _subscribers.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/episodes")
async def episodes():
    eps = sorted(config.EPISODES.iterdir(),
                 key=lambda p: (p / "episode.json").stat().st_mtime if (p / "episode.json").exists() else 0,
                 reverse=True)
    out = []
    for p in eps:
        m = p / "episode.json"
        if m.exists():
            out.append(json.loads(m.read_text())["id"])
    return {"episodes": out}


@app.get("/")
async def index():
    return FileResponse(_STATIC / "index.html")


if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8400, log_level="info")
