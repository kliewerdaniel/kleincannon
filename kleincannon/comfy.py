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


# ---------------------------------------------------------------- workflow
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
    path = path or config.COMFY_WORKFLOW
    return ui_to_api(json.loads(path.read_text()))


def workflow_settings(path: Path | None = None) -> dict:
    """Read the workflow's native render settings straight from the UI file."""
    ui = json.loads((path or config.COMFY_WORKFLOW).read_text())
    out: dict = {k: None for k in ("steps", "cfg", "width", "height")}
    for node in ui.get("nodes", []):
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
                 width: int | None = None, height: int | None = None) -> dict:
    """Patch the klein graph: prompts, seed, latent size."""
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

    # width/height — two PrimitiveInt nodes feed the latent + scheduler.
    # We set ALL PrimitiveInt widgets that are currently the workflow's dims.
    if width is not None and height is not None:
        prims = _by_type(graph, N_PRIMITIVE)
        for nid in prims:
            wv = graph[nid].get("widgets_values") or []
            # PrimitiveInt wv = [value, control]; set value if it looks like a dim
            if wv and isinstance(wv[0], int):
                # width appears once, height once in the original; match by value
                if wv[0] == config.GEN_WIDTH and width != config.GEN_WIDTH:
                    wv[0] = width
                elif wv[0] == config.GEN_HEIGHT and height != config.GEN_HEIGHT:
                    wv[0] = height
                elif wv[0] not in (config.GEN_WIDTH, config.GEN_HEIGHT):
                    # already a non-default dim from a custom workflow; leave it
                    pass
                graph[nid]["widgets_values"] = wv
        # also force-set literal nodes if present
        for nid in _by_type(graph, N_LATENT):
            graph[nid]["inputs"]["width"] = width
            graph[nid]["inputs"]["height"] = height
        for nid in _by_type(graph, N_SCHED):
            graph[nid]["inputs"]["width"] = width
            graph[nid]["inputs"]["height"] = height

    # Bypass the Lora Loader if present (user reported the LoRA is not needed and
    # its file may be absent). mode=4 == bypass in litegraph.
    for nid in _by_type(graph, N_LORA):
        # api graph has no 'mode'; ComfyUI skips bypassed nodes only in UI format.
        # We instead rewire the CFGGuider/CLIP to the upstream model by removing
        # the lora node's effect: easiest is to leave it — if the LoRA file is
        # missing, ComfyUI errors. So we DROP the lora node and reconnect its
        # inputs' source directly. The klein Lora Loader has a single model+clip
        # input (link 180 model, 182 clip). We reconnect those targets to the
        # UnetLoaderGGUF / CLIPLoader outputs instead.
        _drop_lora(graph, nid)

    return graph


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
def is_up() -> bool:
    try:
        urllib.request.urlopen(config.COMFY_URL + "/system_stats", timeout=3)
        return True
    except Exception:
        return False


def ensure_server(wait: int = 300) -> subprocess.Popen | None:
    """Start ComfyUI from config.COMFY_DIR if it isn't already running.

    Returns the Popen if we started it, None if we reused an existing server.
    """
    if is_up():
        print("[comfy] reusing running ComfyUI")
        return None
    if not config.COMFY_PYTHON.exists():
        raise SystemExit(f"ComfyUI not found at {config.COMFY_PYTHON}")

    python = "/opt/homebrew/bin/python3"
    # CRITICAL: strip PYTHONPATH/PYTHONHOME so the agent venv (py3.11) doesn't
    # leak numpy/torch into ComfyUI's py3.14 process.
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    print(f"[comfy] launching ComfyUI ({config.COMFY_URL}) …")
    proc = subprocess.Popen(
        [python, str(config.COMFY_PYTHON), "--listen", "127.0.0.1", "--port", "8188"],
        cwd=str(config.COMFY_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
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


def wait(prompt_id: str, timeout: int = 5400, poll: float = 2.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with urllib.request.urlopen(
            f"{config.COMFY_URL}/history/{prompt_id}", timeout=15
        ) as r:
            hist = json.loads(r.read())
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise SystemExit(f"ComfyUI job failed: {json.dumps(status)[:800]}")
            if status.get("completed") or entry.get("outputs"):
                return entry
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


def generate(positive: str, negative: str, seed: int, dest: Path,
             workflow: dict | None = None,
             width: int | None = None, height: int | None = None) -> Path:
    graph = build_prompt(workflow or load_workflow(), positive, negative, seed,
                          width=width, height=height)
    entry = wait(queue(graph))
    return fetch_images(entry, dest)[0]
