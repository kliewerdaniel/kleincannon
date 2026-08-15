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
from fastapi.responses import FileResponse, JSONResponse
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
    speed: float = config.DEFAULT_TTS_SPEED
    style: str = "auto"              # catalog name | custom suffix | "auto"
    steps: int | None = None
    cfg: float | None = None
    seed: int | None = None
    use_knowledge: bool = False    # P1: ground content in the Obladaet engine


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
        "PROMPT_STYLE": p.style or "auto",
        "IMAGE_SEED": p.seed,
        "IMAGE_STEPS": p.steps,
        "IMAGE_CFG": p.cfg,
        "USE_KNOWLEDGE_ENGINE": bool(p.use_knowledge),
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

        stage("tts", f"Voicing with '{p.voice}' (Qwen3-TTS)")
        # honour the speed slider by writing it onto the episode before voicing
        ep.speed = p.speed or config.DEFAULT_TTS_SPEED
        ep.save()
        tts.run(eid)

        stage("align", "Aligning words to audio")
        align.run(eid)

        stage("prompts", "Composing image prompts")
        prompts.run(eid, style_suffix=p.style or "auto")

        stage("images", "Rendering visuals (ZImage Turbo via ComfyUI)")
        images.run(eid, fast=p.fast,
                   seed=config.IMAGE_SEED, steps=config.IMAGE_STEPS, cfg=config.IMAGE_CFG)

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


@app.get("/api/styles")
async def styles():
    return {
        "default_speed": config.DEFAULT_TTS_SPEED,
        "styles": [
            {"name": s["name"], "palette": s.get("palette", "")}
            for s in config.STYLE_CATALOG
        ],
    }


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


# ---- knowledge-engine API (P1) --------------------------------------------
@app.get("/api/knowledge/status")
async def knowledge_status():
    from kleincannon import knowledge as kb
    return {
        "available": kb.is_available(),
        "enabled": bool(config.USE_KNOWLEDGE_ENGINE),
        "index_dir": str(kb.KNOWLEDGE_INDEX_DIR),
    }


@app.post("/api/knowledge/compile")
async def knowledge_compile(req: Request):
    from kleincannon import knowledge as kb
    body = await req.json()
    roots = body.get("roots")
    return kb.ensure_compiled(roots)
@app.get("/api/learn/status")
async def learn_status():
    from kleincannon.learn import agency
    return agency.status()


@app.get("/api/learn/recommend")
async def learn_recommend(episode: str):
    from kleincannon.learn import optimize
    from kleincannon.episode import Episode
    ep = Episode.load(episode)
    cands = optimize.optimize(ep)
    return {"episode": episode,
            "best": {"variation": cands[0].get("variation"),
                     "predicted_reward": cands[0].get("_predicted_reward"),
                     "uncertainty": cands[0].get("_uncertainty")},
            "top5": [{"variation": c.get("variation"),
                      "predicted_reward": c.get("_predicted_reward")} for c in cands[:5]]}


@app.post("/api/learn/cycle")
async def learn_cycle(req: Request):
    from kleincannon.learn import agency
    from kleincannon.learn import learn_config
    body = await req.json()
    if body.get("auto"):
        learn_config.manual_upload = False
    elif body.get("manual"):
        learn_config.manual_upload = True
    out = agency.run_cycle(
        body["episode"], niche=body.get("niche", ""),
        hashtags=[h.strip() for h in (body.get("hashtags") or "").split(",") if h.strip()],
        caption=body.get("caption", ""), parent_id=body.get("parent"),
        privacy=body.get("privacy"), auto_train=not body.get("no_train", False))
    return out


@app.get("/api/learn/pending")
async def learn_pending():
    from kleincannon.learn import agency
    return agency.pending_packages()


@app.post("/api/learn/record")
async def learn_record(req: Request):
    from kleincannon.learn import agency
    body = await req.json()
    exp_id = body.get("experience_id") or body.get("id")
    if not exp_id:
        return JSONResponse(status_code=400, content={"error": "experience_id required"})
    metrics = {k: float(v) for k, v in (body.get("metrics") or {}).items()
               if isinstance(v, (int, float)) and v}
    if not metrics:
        return JSONResponse(status_code=400, content={"error": "no metrics provided"})
    return agency.record_metrics(
        exp_id, metrics,
        mark_uploaded=not body.get("no_mark_uploaded", False),
        video_id=body.get("video_id"))


# ---- P2: closed-deal attribution API --------------------------------------
@app.post("/api/learn/deal")
async def learn_deal(req: Request):
    from kleincannon.learn import agency
    body = await req.json()
    deal_id = body.get("deal_id")
    value = body.get("value")
    if not deal_id or not isinstance(value, (int, float)) or value <= 0:
        return JSONResponse(status_code=400,
                            content={"error": "deal_id and positive value required"})
    return agency.record_deal(
        deal_id=str(deal_id), value=float(value), offer=body.get("offer", ""),
        touchpoints=body.get("touchpoints"),
        attributed_experience_ids=body.get("attributed_experience_ids"),
        attribution_method=body.get("attribution_method"),
        confidence=float(body.get("confidence", 1.0)),
        source_ref=body.get("source_ref", ""))


@app.get("/api/learn/conversions")
async def learn_conversions():
    from kleincannon.learn import attribution as attr
    return {"conversions": attr.list_deals(),
            "attributed_total": round(attr.attributed_total(), 5)}


@app.get("/api/learn/attribution/preview")
async def learn_attribution_preview(deal_id: str):
    from kleincannon.learn import db
    store = db.open_db()
    deal = store.get_conversion(deal_id)
    store.close()
    if not deal:
        return JSONResponse(status_code=404, content={"error": "no such deal"})
    from kleincannon.learn import attribution as attr
    return {"deal_id": deal_id, "credits": attr.credit_for(deal)}


@app.post("/api/learn/harvest")
async def learn_harvest():
    from kleincannon.learn import harvester as hv
    return hv.harvest_all()


@app.post("/api/learn/train")
async def learn_train():
    from kleincannon.learn import trainer as tr
    return tr.train_from_history()


@app.get("/api/learn/history")
async def learn_history():
    from kleincannon.learn import trainer as tr
    return tr.history()


@app.get("/learn")
async def learn_page():
    return FileResponse(_STATIC / "learn.html")


@app.get("/")
async def index():
    return FileResponse(_STATIC / "index.html")


if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8400, log_level="info")
