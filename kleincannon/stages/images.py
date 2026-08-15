"""Stage 5 — render the configured ComfyUI workflow's images (N shots per beat).

Drives the in-repo workflow (ZImage Turbo). ComfyUI is auto-launched if needed
(see comfy.ensure_server). Each shot is written to images/<beat>_s<j>.png and the
filenames are recorded on the beat (b.images) so assemble.py / captions.py can
find them.

Resumable: shots whose image already exists are skipped, so an interrupted run
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
        cfg: float | None = None,
        shots_per_beat: int = config.SHOTS_PER_BEAT) -> Episode:
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

    import random
    for i, beat in enumerate(ep.beats):
        if not beat.shots:
            # legacy single-prompt path: treat image_prompt as one shot
            if beat.image_prompt:
                beat.shots = [beat.image_prompt]
            else:
                continue
        rendered: list[str] = []
        for j, prompt in enumerate(beat.shots):
            dest = ep.images_dir / f"{beat.id}_s{j + 1}.png"
            if dest.exists() and not force and not stale_provider:
                print(f"  {beat.id}.shot{j + 1}  skip (exists)")
                rendered.append(f"images/{beat.id}_s{j + 1}.png")
                continue
            if dest.exists() and (force or stale_provider):
                print(f"  {beat.id}.shot{j + 1}  re-render ({('forced' if force else 'workflow changed')})")
            # A forced re-render must NOT reuse the deterministic cache seed, or
            # ComfyUI's execution cache would serve the previous render for this
            # (graph + seed) pair instead of actually recomputing. Use a fresh
            # random seed so the cache can never hit.
            s = random.randint(1, 2_147_483_646)
            print(f"  {beat.id}.shot{j + 1}  rendering (seed {s}) …")
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
                        positive=prompt,
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
                raise SystemExit(f"image stage failed for {beat.id}.shot{j + 1} after {max_attempts} attempts")
            rendered.append(f"images/{beat.id}_s{j + 1}.png")
            print(f"  {beat.id}.shot{j + 1}  -> {dest.name}")
        beat.images = rendered
        beat.image = rendered[0] if rendered else beat.image

    ep.image_workflow = workflow_name
    ep.save()
    done = sum(1 for b in ep.beats if b.images and (ep.images_dir / b.images[0]).exists())
    total_shots = sum(len(b.all_images) for b in ep.beats)
    print(f"[images] {done}/{len(ep.beats)} beats, {total_shots} shots rendered")
    return ep
