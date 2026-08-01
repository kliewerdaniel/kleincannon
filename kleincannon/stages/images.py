"""Stage 5 — render one image per beat with FLUX.2-klein via ComfyUI.

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
        seed: int | None = None) -> Episode:
    ep = Episode.load(episode_id)
    if not any(b.image_prompt for b in ep.beats):
        raise SystemExit("run the prompts stage first (no image prompts)")

    # Make sure ComfyUI is up (auto-launches if needed; strips env vars).
    comfy.ensure_server()

    workflow = comfy.load_workflow()
    w, h = _dims(fast)
    print(f"[images] FLUX.2-klein  {w}x{h}  fast={fast}  "
          f"(native {config.GEN_WIDTH}x{config.GEN_HEIGHT})")

    for i, beat in enumerate(ep.beats):
        if not beat.image_prompt:
            continue
        dest = ep.images_dir / f"{beat.id}.png"
        if dest.exists() and not force:
            print(f"  {beat.id}  skip (exists)")
            beat.image = f"images/{beat.id}.png"
            continue

        s = seed if seed is not None else 1000 + i * 137
        print(f"  {beat.id}  rendering (seed {s}) …")
        try:
            comfy.generate(
                positive=beat.image_prompt,
                negative=NEG,
                seed=s,
                dest=dest,
                workflow=workflow,
                width=w,
                height=h,
            )
        except SystemExit as e:
            # Surface a friendly note about the missing-LoRA situation if relevant.
            raise
        beat.image = f"images/{beat.id}.png"
        print(f"  {beat.id}  -> {dest.name}")

    ep.save()
    done = sum(1 for b in ep.beats if (ep.images_dir / f"{b.id}.png").exists())
    print(f"[images] {done}/{len(ep.beats)} beats rendered")
    return ep
