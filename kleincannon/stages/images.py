"""Stage 5 — render one image per beat with the configured ComfyUI workflow.

Drives the in-repo klein.json workflow. ComfyUI is auto-launched if needed
(see comfy.ensure_server). Images are written to images/<beat>.png and the
filename is recorded on the beat so assemble.py / captions.py can find it.

Resumable: beats whose image already exists are skipped, so an interrupted run
(or a single-beat re-render) only does the missing work.
"""
from __future__ import annotations

from pathlib import Path

from .. import comfy, config
from ..episode import Episode
from ..stages import prompts as prompts_stage

NEG = prompts_stage.NEGATIVE


def _dims(fast: bool) -> tuple[int, int]:
    if fast:
        return config.FAST_GEN_WIDTH, config.FAST_GEN_HEIGHT
    return config.GEN_WIDTH, config.GEN_HEIGHT


def run(episode_id: str, fast: bool = False, force: bool = False,
        seed: int | None = None, steps: int | None = None,
        cfg: float | None = None) -> Episode:
    ep = Episode.load(episode_id)
    if not any(b.image_prompt for b in ep.beats):
        raise SystemExit("run the prompts stage first (no image prompts)")

    # Make sure ComfyUI is up (auto-launches if needed; strips env vars).
    comfy.ensure_server()

    workflow = comfy.load_workflow()
    workflow_name = config.COMFY_WORKFLOW.name  # e.g. zimageturbo.json
    w, h = _dims(fast)
    print(f"[images] ZImage Turbo  {w}x{h}  fast={fast}  "
          f"(native {config.GEN_WIDTH}x{config.GEN_HEIGHT})")

    # A provider / workflow change means stale images from a previous run must be
    # re-rendered even when --force isn't set (otherwise a `kc all` reuses old
    # FLUX.2-klein images after a swap to ZImage Turbo). The orchestrator's
    # `kc all` path also passes force=True so every beat is always fresh.
    stale_provider = (ep.image_workflow is not None
                      and ep.image_workflow != workflow_name)

    for i, beat in enumerate(ep.beats):
        if not beat.image_prompt:
            continue
        dest = ep.images_dir / f"{beat.id}.png"
        if dest.exists() and not force and not stale_provider:
            print(f"  {beat.id}  skip (exists)")
            beat.image = f"images/{beat.id}.png"
            continue
        if dest.exists() and (force or stale_provider):
            reason = "forced" if force else f"workflow changed ({ep.image_workflow} -> {workflow_name})"
            print(f"  {beat.id}  re-render ({reason})")

        s = seed if seed is not None else 1000 + i * 137
        if force or stale_provider:
            # A forced re-render must NOT reuse the deterministic cache seed, or
            # ComfyUI's execution cache would serve the previous render for this
            # (graph + seed) pair instead of actually recomputing. Use a fresh
            # random seed so the cache can never hit.
            import random
            s = random.randint(1, 2_147_483_646)
        print(f"  {beat.id}  rendering (seed {s}) …")
        # ComfyUI is run by the user (COMFY_AUTO_LAUNCH=False): one long-lived
        # server on :8188, never relaunched between beats (relaunching can poison
        # the Apple-Silicon MPS pool). We just queue the job and wait for the save
        # node's file to land on disk. If a connection drops we retry against the
        # SAME server rather than killing it.
        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            if not comfy.is_up():
                comfy.ensure_server()
            try:
                comfy.generate(
                    positive=beat.image_prompt,
                    negative=NEG,
                    seed=s,
                    dest=dest,
                    workflow=workflow,
                    width=w,
                    height=h,
                    steps=steps,
                    cfg=cfg,
                )
            except (comfy.ComfyServerDied, SystemExit) as e:
                print(f"    attempt {attempt} failed ({e}); retrying on same server")
            if dest.exists() and dest.stat().st_size > 0:
                break
        if not (dest.exists() and dest.stat().st_size > 0):
            raise SystemExit(f"image stage failed for {beat.id} after {max_attempts} attempts")
        beat.image = f"images/{beat.id}.png"
        print(f"  {beat.id}  -> {dest.name}")

    ep.image_workflow = workflow_name
    ep.save()
    done = sum(1 for b in ep.beats if (ep.images_dir / f"{b.id}.png").exists())
    print(f"[images] {done}/{len(ep.beats)} beats rendered")
    return ep
