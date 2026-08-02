"""Stage 2 — voice the script with Qwen3-TTS (MLX, server-less).

Qwen3-TTS is an in-context-learning (ICL) voice-cloning model: given a short
reference clip plus the (whisper-transcribed) transcript of that clip, it speaks
any text in that voice. We drive it through `mlx-audio` as a one-shot in-process
call, so there is **no long-lived TTS server**.

This is the same engine used by the companion `vox` app; we vendor the approach
here so the whole pipeline lives in one folder.

Long scripts are split on sentence boundaries; each chunk is synthesized
separately and the resulting WAVs are concatenated into one `voice.wav`.
Because Qwen3-TTS does not pad clauses with trailing silence the way F5-TTS
does, the chunks abut almost seamlessly — we only trim genuine leading/trailing
room-tone and cross-fade the joins by a few ms so there are no clicks and no
audible dead-air pauses, while still sounding natural (not machine-cut).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import config
from ..episode import Episode

# Reference clips live in voices/<name>.wav (or .mp3/.flac/.m4a/.ogg).
VOICE_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")

# Qwen3-TTS base model (4-bit MLX). Supports ICL zero-shot cloning via
# ref_audio + ref_text. ~1.5 GB download, cached in the HF hub cache.
QWEN3_MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit"

# Whisper model used to auto-transcribe reference clips (no manual .txt needed).
REF_STT_MODEL = "base"

# How much leading/trailing room-tone we may trim from each clause (seconds).
# Qwen clips are already tight, so we keep this small and gentle.
MAX_TRIM = 0.6
# Amplitude floor (linear) below which we treat audio as silence. -50 dBFS.
SILENCE_FLOOR = 10 ** (-50 / 20)
# Tiny cross-fade between concatenated chunks so joins are click-free but
# inaudible (the speech itself is NOT cross-faded, only the ~12ms boundary).
JOIN_FADE = 0.012


def _resolve_voice(ep: Episode) -> Path:
    """Return the reference clip path for the episode's voice name."""
    base = config.VOICES / ep.voice
    clip = None
    for ext in VOICE_EXTS:
        cand = base.with_suffix(ext)
        if cand.exists():
            clip = cand
            break
    if clip is None and base.is_dir():
        for f in sorted(base.iterdir()):
            if f.suffix.lower() in VOICE_EXTS:
                clip = f
                break
    if clip is None:
        raise SystemExit(
            f"voice sample missing: {base} (.wav/.mp3/.flac/.m4a/.ogg or dir/)")
    return clip


def _chunks(text: str, max_words: int = 50) -> list[str]:
    """Split into sentence-ish chunks Qwen can handle in one call.

    Splits on sentence boundaries (keeping the delimiter) so natural pauses are
    preserved; caps at ~50 words/chunk (~20s at normal speech rate).
    """
    parts = re.split(r"(?<=[.!?;:])\s+", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return [text] if text.strip() else []
    chunks, cur, n = [], "", 0
    for p in parts:
        w = len(p.split())
        if n + w > max_words and cur:
            chunks.append(cur)
            cur, n = p, w
        else:
            cur = (cur + " " + p).strip()
            n += w
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c.strip()]


def _transcribe_ref(clip: Path) -> str:
    """Transcribe the reference clip with faster-whisper so Qwen's ref_text
    EXACTLY matches the audio. Cached to a sidecar so we don't re-transcribe
    every run. Falls back to '' if whisper is unavailable.
    """
    cache = clip.with_name(clip.stem + ".reftext")
    try:
        if cache.exists() and cache.stat().st_mtime >= clip.stat().st_mtime:
            return cache.read_text().strip()
    except OSError:
        pass
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return ""
    model = WhisperModel(REF_STT_MODEL, device="cpu", compute_type="int8")
    segs, _ = model.transcribe(str(clip), beam_size=4)
    text = " ".join(s.text for s in segs).strip()
    try:
        cache.write_text(text)
    except OSError:
        pass
    return text


def _load_model():
    from mlx_audio.tts.utils import load_model
    return load_model(QWEN3_MODEL_ID)


def _synth_chunk(model, text: str, ref_audio: Path | None, ref_text: str | None,
                 speed: float) -> np.ndarray:
    """Generate one chunk; return float32 mono audio at 24000 Hz."""
    sr = 24000
    parts = []
    kwargs = dict(text=text, speed=speed, verbose=False)
    if ref_audio is not None:
        kwargs["ref_audio"] = str(ref_audio)
        if ref_text:
            kwargs["ref_text"] = ref_text
    for r in model.generate(**kwargs):
        parts.append(np.asarray(r.audio, dtype=np.float32))
        sr = getattr(r, "sample_rate", sr)
    if not parts:
        raise SystemExit(f"Qwen produced no audio for chunk: {text[:60]!r}")
    return (np.concatenate(parts) if len(parts) > 1 else parts[0]), sr


def _trim_roomtone(audio: np.ndarray, sr: int) -> np.ndarray:
    """Trim genuine leading/trailing silence, capped at MAX_TRIM per end.

    Uses a gentle -50 dBFS floor and only removes clearly-empty room-tone, so
    the speech itself is never cut. Returns the trimmed array.
    """
    n = len(audio)
    if n == 0:
        return audio
    above = np.abs(audio) > SILENCE_FLOOR
    idx = np.nonzero(above)[0]
    if len(idx) == 0:
        return audio
    first, last = int(idx[0]), int(idx[-1])
    first = min(first, int(MAX_TRIM * sr))
    tail = min(n - 1 - last, int(MAX_TRIM * sr))
    return audio[first:n - tail]


def _crossfade(a: np.ndarray, b: np.ndarray, sr: int) -> np.ndarray:
    """Join two chunks with a tiny equal-power cross-fade (click-free, inaudible)."""
    fade = max(1, int(JOIN_FADE * sr))
    if len(a) < fade * 2 or len(b) < fade * 2:
        return np.concatenate([a, b])
    head, tail_a = a[:-fade], a[-fade:]
    tail, body_b = b[:fade], b[fade:]
    t = np.linspace(0.0, 1.0, fade)
    cross = tail_a * (1 - t) + tail * t
    return np.concatenate([head, cross, body_b])


def run(episode_id: str, speed: float = 1.0, voice: str | None = None) -> Episode:
    ep = Episode.load(episode_id)
    if voice:
        ep.voice = voice
        ep.save()
    if not ep.voice:
        ep.voice = config.DEFAULT_VOICE
        ep.save()

    clip = _resolve_voice(ep)
    text = ep.full_script
    if not text:
        raise SystemExit("episode has no script text to voice")

    chunks = _chunks(text)
    print(f"[tts] voicing {len(chunks)} chunk(s) as '{ep.voice}' "
          f"({len(text.split())} words) via Qwen3-TTS …")

    ref_text = _transcribe_ref(clip) if clip is not None else None
    if ref_text:
        print(f"[tts] ref transcript: {ref_text[:80]!r}")
    else:
        print("[tts] no whisper transcript — using default (uncloned) voice")

    model = _load_model()
    ep.mkdirs()
    sr = 24000
    segments: list[np.ndarray] = []
    try:
        for i, chunk in enumerate(chunks):
            audio, csr = _synth_chunk(model, chunk, clip, ref_text, speed)
            sr = csr
            audio = _trim_roomtone(audio, sr)
            segments.append(audio)
            print(f"[tts]   chunk {i+1}/{len(chunks)} {len(audio)/sr:.2f}s")
    except Exception:
        # model is in-process; nothing to clean up beyond the audio we drop
        raise

    # Concatenate with inaudible cross-fades (no dead-air pauses, no clicks).
    merged = segments[0]
    for seg in segments[1:]:
        merged = _crossfade(merged, seg, sr)

    dest = ep.audio_dir / "voice.wav"
    sf.write(str(dest), merged, sr)
    ep.voice_audio = "audio/voice.wav"
    ep.save()
    print(f"[tts] -> {dest}  ({len(merged)/sr:.2f}s, {sr} Hz)")
    return ep
