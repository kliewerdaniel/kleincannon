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

KENBURN_FRAMES = 90   # used only as a floor; real motion spans the full beat


def _probe_image_size(ep: Episode) -> tuple[int, int]:
    for b in ep.beats:
        if b.image:
            p = ep.dir / b.image
            if p.exists():
                from PIL import Image
                with Image.open(p) as im:
                    return im.width, im.height
    return config.GEN_WIDTH, config.GEN_HEIGHT


def _kenburns(beat, w: int, h: int, beat_frames: float) -> str:
    """Continuous Ken Burns for one beat.

    Motion spans the WHOLE beat duration (no static hold), so there is never a
    frozen frame. Beats alternate push-in / push-out so a cut lands on a moving
    frame at a different zoom — keeping motion continuous across the edit.
    """
    n = max(KENBURN_FRAMES, int(round(beat_frames)))
    z = config.ZOOM_MAX
    if beat.motion == "out":
        # start zoomed in (z), ease back out to 1.0 by the end
        zp = f"min({z},max(1.0,(1.0+({z}-1.0)*(1-(on-1)/{n}))))"
        x_expr, y_expr = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif beat.motion == "left":
        x_expr = f"iw/zoom*((on-1)/{n})"
        y_expr = "ih/2-(ih/zoom/2)"
        zp = str(z)
    elif beat.motion == "right":
        x_expr = f"iw/zoom*(1-(on-1)/{n})"
        y_expr = "ih/2-(ih/zoom/2)"
        zp = str(z)
    else:  # "in" — slow continuous push-in across the whole beat
        zp = f"min({z},max(1.0,1.0+({z}-1.0)*((on-1)/{n})))"
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
        beat_frames = dur * config.FPS
        kb = _kenburns(b, w, h, beat_frames)
        filters.append(
            f"[{i}:v]trim=duration={dur:.3f},setpts=PTS-STARTPTS,"
            f"fps={config.FPS},{kb},format=yuv420p[v{i}]"
        )
    vcat = "".join(f"[v{i}]" for i, b in enumerate(ep.beats) if b.image)
    filters.append(f"{vcat}concat=n={len(img_inputs)}:v=1:a=0[vcat]")

    # Caption overlay — single caption-layer video (one transparent PNG per
    # frame, karaoke-highlighted), composited with ONE overlay filter. This
    # avoids ffmpeg's silent truncation of long sequential overlay chains.
    cap_json = ep.dir / "captions" / "words.json"
    frames_dir = ep.dir / "captions" / "frames"
    if cap_json.exists() and frames_dir.exists():
        meta = json.loads(cap_json.read_text())
        n_frames = meta["n_frames"]
        fps = meta.get("fps", config.FPS)
        # Build the caption-layer video from the frame PNGs (glob order is
        # frame_00000.png .. frame_NNNNN.png).
        cap_vid = ep.dir / "captions" / "caption_layer.mp4"
        fr = str(frames_dir / "frame_%05d.png")
        cl = [
            config.FFMPEG, "-y", "-hide_banner",
            "-framerate", str(fps), "-start_number", "0",
            "-i", fr,
            "-frames:v", str(n_frames),
            "-c:v", "png",  # lossless, keep alpha
            str(cap_vid),
        ]
        subprocess.run(cl, capture_output=True, text=True, check=True)
        cmd += ["-i", str(cap_vid)]
        # one overlay of the full-frame transparent layer over the concat
        cap_label = f"{len(img_inputs) + (1 if voice and voice.exists() else 0)}:v"
        filters.append(f"[vcat][{cap_label}]overlay=format=auto[vout]")
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
    # Manifest must reflect the current voice.wav length — otherwise the build
    # trims to a stale duration and the narration is cut off mid-sentence.
    import kleincannon.stages.align as align_stage
    if align_stage.needs_realign(ep):
        print("[assemble] voice.wav newer than manifest — realigning first …")
        align_stage.run(episode_id)
        ep = Episode.load(episode_id)
    # The caption frames must be newer than the aligned manifest + audio —
    # otherwise we'd composite karaoke timed to a *different* TTS run, which
    # desyncs from the speech and drops the final words. Re-render if stale.
    import kleincannon.stages.captions as captions_stage
    if align_stage.needs_recaption(ep):
        print("[assemble] caption frames stale — re-rendering captions …")
        captions_stage.run(episode_id)
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
