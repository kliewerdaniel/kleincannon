"""ComfyUI client for the FLUX.2-klein GGUF workflow.

Loads the in-repo UI workflow (klein.json), converts it to the /prompt API graph,
patches the prompt text / seed / latent size, queues it, polls /history, and
copies the result out.

ComfyUI is the only remaining long-lived service. `ensure_server()` starts it
from the repo config if it isn't already running, so `kc images` "just works"
without you launching anything. (If a server is already up on COMFY_URL we reuse it.)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from . import config

CLIENT_ID = str(uuid.uuid4())

# Node types in the klein workflow we patch by.
N_POS = "CLIPTextEncode"          # positive prompt (first CLIPTextEncode)
N_NEG = "CLIPTextEncode"          # negative prompt (the empty one)
N_LATENT = "EmptyFlux2LatentImage"
N_SCHED = "Flux2Scheduler"
N_NOISE = "RandomNoise"
N_PRIMITIVE = "PrimitiveInt"      # width/height (two of them: width then height)
N_LORA = "Lora Loader (LoraManager)"
N_CFG = "CFGGuider"               # cfg scale


# ---------------------------------------------------------------- workflow
def _looks_like_api_graph(obj: object) -> bool:
    """True if `obj` is already an API-format prompt graph.

    API graphs are dicts mapping node-id -> {"class_type": str, "inputs": {...}}.
    UI (litegraph) files instead have top-level keys like "nodes"/"links".
    """
    if not isinstance(obj, dict):
        return False
    if "nodes" in obj or "links" in obj:
        return False
    # Every value should look like an API node.
    return all(
        isinstance(v, dict) and "class_type" in v and "inputs" in v
        for v in obj.values()
    )


def ui_to_api(ui: dict) -> dict:
    """Convert a saved litegraph UI workflow into the /prompt API graph."""
    link_src = {}
    for link in ui.get("links", []):
        # [id, origin_node, origin_slot, target_node, target_slot, type]
        link_src[link[0]] = [str(link[1]), link[2]]

    api: dict[str, dict] = {}
    for node in ui["nodes"]:
        if node.get("mode") in (2, 4):      # muted / bypassed
            continue
        ntype = node["type"]
        if ntype in ("Note", "MarkdownNote", "Reroute", "PrimitiveNode"):
            continue

        inputs: dict = {}
        widget_vals = list(node.get("widgets_values") or [])
        wi = 0
        for inp in node.get("inputs", []):
            name = inp["name"]
            if inp.get("link") is not None:
                inputs[name] = link_src[inp["link"]]
            elif "widget" in inp:
                if wi < len(widget_vals):
                    inputs[name] = widget_vals[wi]
                    wi += 1
                    if (name in ("seed", "noise_seed", "value")
                            and wi < len(widget_vals)
                            and isinstance(widget_vals[wi], str)):
                        wi += 1   # skip the control_after_generate value
        api[str(node["id"])] = {
            "class_type": ntype,
            "inputs": inputs,
            "_meta": {"title": node.get("title", ntype)},
        }
    return api


def load_workflow(path: Path | None = None) -> dict:
    """Load the klein workflow as an API-format prompt graph.

    If the file is already an API-format graph (dict of node-id -> {class_type,
    inputs}), it is used directly. This is the preferred form: the committed
    klein.json is the exact graph that is known to render on this machine, so we
    avoid the lossy UI->API conversion that mis-assigns widget values (e.g.
    EmptyFlux2LatentImage batch_size). UI-format litegraph files are converted
    via ui_to_api() as a fallback.
    """
    path = path or config.COMFY_WORKFLOW
    raw = json.loads(path.read_text())
    if _looks_like_api_graph(raw):
        return raw
    return ui_to_api(raw)


def workflow_settings(path: Path | None = None) -> dict:
    """Read the workflow's native render settings.

    Handles both API-format graphs (committed klein.json) and UI litegraph files:
    dims come from the EmptyFlux2LatentImage inputs (or the PrimitiveInt width/
    height nodes), steps from Flux2Scheduler, cfg from CFGGuider.
    """
    raw = json.loads((path or config.COMFY_WORKFLOW).read_text())
    if isinstance(raw, dict) and "nodes" not in raw and "links" not in raw:
        # API-format graph
        width = height = steps = cfg = None
        for n in raw.values():
            ct = n.get("class_type")
            if ct == "EmptyFlux2LatentImage":
                width, height = n["inputs"].get("width"), n["inputs"].get("height")
            elif ct == "Flux2Scheduler":
                steps = n["inputs"].get("steps")
            elif ct == "CFGGuider":
                cfg = n["inputs"].get("cfg")
        if width is None:  # fall back to PrimitiveInt width/height nodes
            for n in raw.values():
                if n.get("class_type") == "PrimitiveInt" and "value" in n.get("inputs", {}):
                    v = n["inputs"]["value"]
                    if width is None:
                        width = v
                    else:
                        height = v
        return {"steps": steps, "cfg": cfg, "width": width, "height": height}
    # UI-format litegraph
    out: dict = {k: None for k in ("steps", "cfg", "width", "height")}
    for node in raw.get("nodes", []):
        if node.get("type") == "Flux2Scheduler":
            wv = node.get("widgets_values") or []
            if len(wv) >= 1:
                out["steps"] = wv[0]
        if node.get("type") == "CFGGuider":
            wv = node.get("widgets_values") or []
            if wv:
                out["cfg"] = wv[0]
        if node.get("type") == "EmptyFlux2LatentImage":
            wv = node.get("widgets_values") or []
            if len(wv) >= 2:
                out["width"], out["height"] = wv[0], wv[1]
    return out


def _by_type(api: dict, class_type: str) -> list[str]:
    return [k for k, v in api.items() if v["class_type"] == class_type]


def build_prompt(api: dict, positive: str, negative: str, seed: int,
                 width: int | None = None, height: int | None = None,
                 steps: int | None = None, cfg: float | None = None) -> dict:
    """Patch the klein graph: prompts, seed, latent size, steps, cfg."""
    graph = json.loads(json.dumps(api))

    # positive = the CLIPTextEncode whose text is non-empty in the original;
    # negative = the empty one. Decide by current widget text.
    pos_nodes = _by_type(graph, N_POS)
    pos_id = neg_id = None
    for nid in pos_nodes:
        txt = graph[nid]["inputs"].get("text", "")
        if isinstance(txt, str) and txt.strip():
            pos_id = nid
        else:
            neg_id = nid
    if pos_id is None and pos_nodes:
        pos_id = pos_nodes[0]
    if neg_id is None and len(pos_nodes) > 1:
        neg_id = [n for n in pos_nodes if n != pos_id][0]
    if pos_id:
        graph[pos_id]["inputs"]["text"] = positive
    if neg_id:
        graph[neg_id]["inputs"]["text"] = negative

    # seed
    noise = _by_type(graph, N_NOISE)
    if noise:
        graph[noise[0]]["inputs"]["noise_seed"] = seed

    # width/height — the working klein graph sets these via two PrimitiveInt
    # nodes (width=1080, height=1920) whose values feed the latent and
    # scheduler through links. We set the PrimitiveInt inputs["value"] so
    # the linked nodes pick up the requested dims; we also force-set the
    # literal nodes (EmptyFlux2LatentImage / Flux2Scheduler) for good measure.
    if width is not None and height is not None:
        prims = _by_type(graph, N_PRIMITIVE)
        for nid in prims:
            cur = graph[nid].get("inputs", {}).get("value")
            if isinstance(cur, int):
                if cur == config.GEN_WIDTH:
                    graph[nid]["inputs"]["value"] = width
                elif cur == config.GEN_HEIGHT:
                    graph[nid]["inputs"]["value"] = height
                elif cur not in (config.GEN_WIDTH, config.GEN_HEIGHT):
                    # already a non-default size from a custom workflow; leave it
                    pass
        for nid in _by_type(graph, N_LATENT):
            graph[nid]["inputs"]["width"] = width
            graph[nid]["inputs"]["height"] = height
        for nid in _by_type(graph, N_SCHED):
            graph[nid]["inputs"]["width"] = width
            graph[nid]["inputs"]["height"] = height

    # steps (Flux2Scheduler) / cfg (CFGGuider) — only if the caller overrides
    if steps is not None:
        for nid in _by_type(graph, N_SCHED):
            if "steps" in graph[nid]["inputs"]:
                graph[nid]["inputs"]["steps"] = steps
    if cfg is not None:
        for nid in _by_type(graph, N_CFG):
            if "cfg" in graph[nid]["inputs"]:
                graph[nid]["inputs"]["cfg"] = cfg

    # Bypass the Lora Loader if present (user reported the LoRA is not needed and
    # its file may be absent). mode=4 == bypass in litegraph.
    for nid in _by_type(graph, N_LORA):
        _drop_lora(graph, nid)

    # Static workflow uses stale model filenames / a custom save node whose
    # widget-only inputs ui_to_api() can't see. Reconcile both with the live
    # ComfyUI server.
    _fix_model_names(graph)
    _patch_save_nodes(graph, seed)
    _drop_memory_cleanup(graph)

    return graph


def _fix_model_names(graph: dict) -> None:
    """Remap the static workflow's model filenames to what the live ComfyUI
    server actually has on disk (strips the 'FLUX.2/' subdir prefix the saved
    UI workflow used, and downgrades the unet from Q8 to the present Q4)."""
    remap = {
        "unet_name": {
            "FLUX.2/flux-2-klein-4b-Q8_0.gguf": "flux-2-klein-4b-Q4_K_M.gguf",
        },
        "clip_name": {
            "FLUX.2/qwen_3_4b.safetensors": "qwen_3_4b.safetensors",
        },
        "vae_name": {
            "FLUX.2/flux2-vae.safetensors": "flux2-vae.safetensors",
        },
    }
    for node in graph.values():
        for field, table in remap.items():
            val = node["inputs"].get(field)
            if isinstance(val, str) and val in table:
                node["inputs"][field] = table[val]


def _patch_save_nodes(graph: dict, seed: int) -> None:
    """The custom 'Image Saver Simple' node exposes its settings as widget-only
    inputs that ui_to_api() drops. Re-inject the required inputs from the live
    /object_info schema so ComfyUI validation passes and writes a named file.

    We route each render into a size-specific subfolder (e.g. "1080x1920") so a
    re-run at a different resolution writes a NON-colliding filename. Without
    this, ComfyUI's execution cache serves the previous run's file (same seed +
    same graph hash) and a native re-render silently returns the old low-res
    image — which is exactly why `--force` produced stale 528x960 frames.
    """
    dims = "render"
    try:
        for node in graph.values():
            if node["class_type"] == "EmptyFlux2LatentImage":
                w = node["inputs"].get("width")
                h = node["inputs"].get("height")
                if w and h:
                    dims = f"{w}x{h}"
                break
    except Exception:
        pass
    for node in graph.values():
        if node["class_type"] != "Image Saver Simple":
            continue
        node["inputs"].update({
            "filename": f"klein_{seed}",
            "path": dims,          # subfolder = render resolution
            "extension": "png",
            "lossless_webp": False,
            "quality_jpeg_or_webp": 95,
            "optimize_png": True,
            "embed_workflow": False,
            "save_workflow_as_json": False,
        })


def _drop_memory_cleanup(graph: dict) -> None:
    """Remove orphaned VRAM/RAM cleanup nodes (comfyui_memory_cleanup custom
    nodes). Their widget-only inputs are dropped by ui_to_api(), ComfyUI
    rejects the graph, and they feed nothing in the generation chain — so
    dropping them is safe and makes the klein graph server-valid."""
    for nid in [k for k, v in graph.items()
                if v["class_type"] in ("RAMCleanup", "VRAMCleanup")]:
        graph.pop(nid, None)


def _drop_lora(graph: dict, lora_id: str) -> None:
    """Remove the Lora Loader node and reconnect its consumers to its inputs'
    source nodes (effectively a no-op LoRA pass)."""
    lora = graph.get(lora_id)
    if not lora:
        return
    # find the upstream model + clip that fed the lora
    model_src = lora["inputs"].get("model")
    clip_src = lora["inputs"].get("clip")
    # rewire every node that used the lora's outputs to the source instead
    for nid, node in graph.items():
        if nid == lora_id:
            continue
        for k, v in list(node["inputs"].items()):
            if isinstance(v, list) and v[0] == lora_id:
                if k == "model" and model_src:
                    node["inputs"][k] = model_src
                elif k in ("clip", "positive", "negative") and clip_src:
                    node["inputs"][k] = clip_src
    graph.pop(lora_id, None)


# ---------------------------------------------------------------- server
# Tracked server process so we can tear it down and relaunch between beats.
_SERVER_PROC: subprocess.Popen | None = None


def is_up() -> bool:
    try:
        urllib.request.urlopen(config.COMFY_URL + "/system_stats", timeout=3)
        return True
    except Exception:
        return False


def kill_server() -> None:
    """Hard-stop a ComfyUI process we launched ourselves (COMFY_AUTO_LAUNCH).

    When COMFY_AUTO_LAUNCH is False the server is run by the user on :8188 and we
    must never kill it — relaunching between beats is what poisons the
    Apple-Silicon MPS pool and makes FLUX.2-klein crash at sampler step 0. In
    that mode this is a no-op so the user's long-lived server survives.
    """
    global _SERVER_PROC
    if not getattr(config, "COMFY_AUTO_LAUNCH", False):
        _SERVER_PROC = None
        return
    if _SERVER_PROC is not None and _SERVER_PROC.poll() is None:
        _SERVER_PROC.terminate()
        try:
            _SERVER_PROC.wait(timeout=20)
        except Exception:
            _SERVER_PROC.kill()
    _SERVER_PROC = None
    # Belt-and-suspenders: any stray listener on the port (only ours).
    try:
        subprocess.run(["pkill", "-f", "main.py --listen"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    # Wait for the port to free up.
    for _ in range(30):
        if not is_up():
            break
        time.sleep(1)


def ensure_server(wait: int = 300) -> subprocess.Popen | None:
    """Verify a ComfyUI is reachable on COMFY_URL.

    The server is expected to be RUN BY THE USER on :8188 (COMFY_AUTO_LAUNCH is
    False). In that mode we never launch one ourselves — we just confirm it's up
    and error clearly if it isn't, so the user knows to start it. (If you set
    COMFY_AUTO_LAUNCH=True the agent will launch it with COMFY_LAUNCH_ARGS.)
    """
    global _SERVER_PROC
    if is_up():
        print("[comfy] reusing running ComfyUI")
        return None
    if not getattr(config, "COMFY_AUTO_LAUNCH", False):
        raise SystemExit(
            f"ComfyUI is not running at {config.COMFY_URL}. Start it yourself "
            f"(e.g. from the image/ venv) and re-run. Set COMFY_AUTO_LAUNCH=True "
            f"in config.py to have the agent launch it."
        )
    if not config.COMFY_PYTHON.exists():
        raise SystemExit(f"ComfyUI not found at {config.COMFY_PYTHON}")

    python = str(config.COMFY_VENV_PYTHON)
    # CRITICAL: strip PYTHONPATH/PYTHONHOME so the agent venv (py3.11) doesn't
    # leak numpy/torch into ComfyUI's py3.14 process.
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    print(f"[comfy] launching ComfyUI ({config.COMFY_URL}) …")
    proc = subprocess.Popen(
        [python, str(config.COMFY_PYTHON), "--listen", "127.0.0.1", "--port", "8188",
         *getattr(config, "COMFY_LAUNCH_ARGS", [])],
        cwd=str(config.COMFY_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _SERVER_PROC = proc
    deadline = time.time() + wait
    while time.time() < deadline:
        if is_up():
            print("[comfy] up")
            return proc
        if proc.poll() is not None:
            raise SystemExit("ComfyUI exited during startup")
        time.sleep(2)
    proc.terminate()
    raise SystemExit("ComfyUI did not come up in time")


def restart_server(wait: int = 300) -> None:
    """Tear down + start a clean ComfyUI between beats (auto-launch mode only).

    When COMFY_AUTO_LAUNCH is False the server is the user's long-lived process;
    relaunching it is exactly what poisons the MPS pool, so this is a no-op and
    the same server handles every beat.
    """
    if not getattr(config, "COMFY_AUTO_LAUNCH", False):
        return
    kill_server()
    ensure_server(wait=wait)


def queue(graph: dict) -> str:
    payload = json.dumps({"prompt": graph, "client_id": CLIENT_ID}).encode()
    req = urllib.request.Request(
        config.COMFY_URL + "/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["prompt_id"]
    except urllib.error.HTTPError as e:
        raise SystemExit(f"ComfyUI rejected the graph:\n{e.read().decode()[:1500]}")


class ComfyServerDied(Exception):
    """Raised when ComfyUI drops the connection mid-generation (klein MPS crash).

    Callers should restart the server and retry — the saved image file may
    already be on disk even though /history is unreachable.
    """


def _save_subfolder(graph: dict | None, seed: int) -> str:
    """Read the save node's subfolder ('path') from the graph, or discover the
    real file via /history on crash. Falls back to a size-named guess."""
    if graph:
        for node in graph.values():
            if node["class_type"] == "Image Saver Simple":
                p = node["inputs"].get("path")
                if isinstance(p, str) and p:
                    return p
    return "render"


def _out_file(seed: int, subfolder: str = "") -> Path:
    # "Image Saver Simple" writes filename=f"klein_{seed}" extension="png".
    # ComfyUI appends a _NNNN counter on collision, so glob for the newest
    # klein_{seed}*.png under the render subfolder.
    base = config.COMFY_OUTPUT / subfolder if subfolder else config.COMFY_OUTPUT
    matches = sorted(base.glob(f"klein_{seed}*.png"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else base / f"klein_{seed}.png"


def wait(prompt_id: str, seed: int, dest: Path, timeout: int = 900,
         poll: float = 2.0) -> bool:
    """Poll /history for completion, but also watch the output folder.

    FLUX.2-klein can crash the server at/after the sampler step. The save node
    writes its file to disk before or during that crash, so we treat the
    presence of the output file as success even if the server then dies.

    Returns True on success (dest copied). Raises ComfyServerDied if the
    connection drops before the file appears; the caller restarts + retries.
    """
    deadline = time.time() + timeout
    sub = _save_subfolder(None, seed)
    while time.time() < deadline:
        # The file on disk is the source of truth — it survives a server crash.
        out = _out_file(seed, sub)
        if out.exists() and out.stat().st_size > 0:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out, dest)
            return True
        # Server still alive? probe /history.
        try:
            with urllib.request.urlopen(
                f"{config.COMFY_URL}/history/{prompt_id}", timeout=15
            ) as r:
                hist = json.loads(r.read())
            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise SystemExit(
                        f"ComfyUI job failed: {json.dumps(status)[:800]}"
                    )
                if status.get("completed") or entry.get("outputs"):
                    saved = fetch_images(entry, dest)
                    if saved:
                        return True
        except (urllib.error.URLError, ConnectionError, TimeoutError,
                http_client_exception()) as e:
            # Connection dropped. If the file is already on disk, still a win.
            out = _out_file(seed, sub)
            if out.exists() and out.stat().st_size > 0:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(out, dest)
                return True
            raise ComfyServerDied(
                f"ComfyUI connection lost during generation: {e}"
            ) from e
        time.sleep(poll)
    raise SystemExit(f"ComfyUI job {prompt_id} timed out")


def fetch_images(entry: dict, dest: Path) -> list[Path]:
    """Copy the result image out of ComfyUI's output dir (or via /view)."""
    saved = []
    for node_out in entry.get("outputs", {}).values():
        for img in node_out.get("images", []):
            if img.get("type") == "temp":
                continue
            src = config.COMFY_OUTPUT / img.get("subfolder", "") / img["filename"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy2(src, dest)
            else:
                q = urllib.parse.urlencode({
                    "filename": img["filename"],
                    "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output"),
                })
                with urllib.request.urlopen(f"{config.COMFY_URL}/view?{q}", timeout=60) as r:
                    dest.write_bytes(r.read())
            saved.append(dest)
            return saved
    raise SystemExit("ComfyUI returned no images")


def http_client_exception():
    """Return the http.client exception type for except-clauses at runtime."""
    import http.client
    return http.client.HTTPException


def generate(positive: str, negative: str, seed: int, dest: Path,
             workflow: dict | None = None,
             width: int | None = None, height: int | None = None,
             steps: int | None = None, cfg: float | None = None) -> Path:
    """Queue one generation. On a server crash, copy whatever file landed and
    return; the caller (images.run) restarts the server and retries until the
    destination exists."""
    graph = build_prompt(workflow or load_workflow(), positive, negative, seed,
                          width=width, height=height, steps=steps, cfg=cfg)
    sub = _save_subfolder(graph, seed)
    pid = queue(graph)
    try:
        wait(pid, seed, dest)
    except ComfyServerDied:
        # The save node may have written the file before the crash.
        out = _out_file(seed, sub)
        if out.exists() and out.stat().st_size > 0 and not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out, dest)
        # Propagate so images.run knows to restart + retry if dest missing.
        if not dest.exists():
            raise
    return dest
