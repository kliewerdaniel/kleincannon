"""Stage 7 — assemble the vertical video: Ken Burns + concat + caption overlay.

No music in kleincannon (the track is just the cloned voice). We:
  1. Ken-Burns each beat image (slow zoom/pan) with ffmpeg zoompan, scaled to
     1080x1920.
  2. Concatenate the per-beat clips in beat order.
  3. Composite the Pillow-rendered karaoke caption PNGs on top (ffmpeg `overlay`
     filtered by word timing — no libass needed).
  4. Mux the voice audio, faststart the mp4.

ffmpeg on this Mac lacks libass, so captions come from the transparent PNGs
written by captions.py, not from a .ass file.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from .. import config
from ..episode import Episode

KENBURN_FRAMES = 90   # smoothing for zoompan (3s of motion @30fps internally)


def _probe_image_size(ep: Episode) -> tuple[int, int]:
    for b in ep.beats:
        if b.image:
            p = ep.dir / b.image
            if p.exists():
                from PIL import Image
                with Image.open(p) as im:
                    return im.width, im.height
    return config.GEN_WIDTH, config.GEN_HEIGHT


def _kenburns(beat, w: int, h: int) -> str:
    """zoompan expression for one beat's motion direction."""
    z = config.ZOOM_MAX
    if beat.motion == "out":
        # start zoomed in, drift out to 1.0
        zp = f"min({z},max(1.0,(1.0+({z}-1.0)*(1-(on-1)/{KENBURN_FRAMES}))))"
        x_expr, y_expr = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif beat.motion == "left":
        x_expr = f"iw/zoom*((on-1)/{KENBURN_FRAMES})"
        y_expr = "ih/2-(ih/zoom/2)"
        zp = str(z)
    elif beat.motion == "right":
        x_expr = f"iw/zoom*(1-(on-1)/{KENBURN_FRAMES})"
        y_expr = "ih/2-(ih/zoom/2)"
        zp = str(z)
    else:  # "in" — classic slow push-in
        zp = f"min({z},max(1.0,1.0+({z}-1.0)*((on-1)/{KENBURN_FRAMES})))"
        x_expr, y_expr = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"

    return (
        f"scale={(w*2)}:{(h*2)}:flags=lanczos,"
        f"zoompan=z='{zp}':d=1:s={w}x{h}:x='{x_expr}':y='{y_expr}',"
        f"scale={config.WIDTH}:{config.HEIGHT}:flags=lanczos"
    )


def _build_command(ep: Episode) -> list[str]:
    w, h = _probe_image_size(ep)
    total = ep.total_duration
    cmd = [config.FFMPEG, "-y", "-hide_banner"]

    # Inputs: one looping image + the audio.
    img_inputs: list[Path] = []
    for b in ep.beats:
        if b.image:
            img_inputs.append(ep.dir / b.image)
    voice = ep.dir / ep.voice_audio if ep.voice_audio else None

    for p in img_inputs:
        cmd += ["-loop", "1", "-i", str(p)]
    if voice and voice.exists():
        cmd += ["-i", str(voice)]

    # Build a [v0]...[vN] chain of kenburns'd, time-scaled stills.
    filters = []
    for i, b in enumerate(ep.beats):
        if not b.image:
            continue
        dur = max(0.5, b.duration)
        kb = _kenburns(b, w, h)
        filters.append(
            f"[{i}:v]trim=duration={dur:.3f},setpts=PTS-STARTPTS,"
            f"fps={config.FPS},{kb},format=yuv420p[v{i}]"
        )
    vcat = "".join(f"[v{i}]" for i, b in enumerate(ep.beats) if b.image)
    filters.append(f"{vcat}concat=n={len(img_inputs)}:v=1:a=0[vcat]")

    # Caption overlay (per word), if captions were generated.
    cap_json = ep.dir / "captions" / "words.json"
    if cap_json.exists():
        meta = json.loads(cap_json.read_text())
        words = meta["words"]
        # add each caption PNG as an input AFTER the images and audio
        cap_start = len(img_inputs) + (1 if voice and voice.exists() else 0)
        for wd in words:
            cmd += ["-i", wd["png"]]
        chain = "[vcat]"
        for j, wd in enumerate(words):
            label = f"cap{j}"
            enable = f"between(t,{wd['start']:.3f},{wd['end']:.3f})"
            x = wd.get("x", 0)
            y = wd.get("y", 0)
            chain += (
                f"[{label}]overlay=format=auto:enable='{enable}':"
                f"x={x}:y={y}"
            )
            if j < len(words) - 1:
                chain += f"[o{j}];[o{j}]"
        chain += "[vout]"
        filters.append(chain)
    else:
        filters.append("[vcat]null[vout]")

    cmd += ["-filter_complex", ";".join(filters)]

    # Map video + audio; speed-correct by trimming to total duration.
    cmd += ["-map", "[vout]"]
    if voice and voice.exists():
        cmd += ["-map", f"{len(img_inputs)}:a"]
    cmd += [
        "-t", f"{total:.3f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", str(config.CRF), "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-r", str(config.FPS),
        str(ep.dir / f"{ep.id}.mp4"),
    ]
    return cmd


def run(episode_id: str) -> Episode:
    ep = Episode.load(episode_id)
    missing = [b.id for b in ep.beats if not (b.image and (ep.dir / b.image).exists())]
    if missing:
        raise SystemExit(f"missing images for beats {missing} — run images first")
    if not ep.voice_audio or not (ep.dir / ep.voice_audio).exists():
        raise SystemExit("missing voice audio — run tts first")

    cmd = _build_command(ep)
    print(f"[assemble] building {ep.id}.mp4 ({config.WIDTH}x{config.HEIGHT} @ {config.FPS}fps) …")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # surface the tail of ffmpeg's stderr for debugging
        raise SystemExit(f"ffmpeg failed:\n{proc.stderr[-2500:]}")

    out = ep.dir / f"{ep.id}.mp4"
    ep.final = f"{ep.id}.mp4"
    ep.save()
    print(f"[assemble] -> {out}")
    return ep
