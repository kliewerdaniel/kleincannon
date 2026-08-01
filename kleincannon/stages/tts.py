"""Stage 2 — voice the script with F5TTS (MLX, server-less).

F5-TTS is a zero-shot voice-cloning model: it needs a short *reference* clip plus
the *transcript* of that clip, then speaks any text in that voice. We invoke it as
a one-shot CLI subprocess, so there is **no long-lived TTS server**.

Primary path: the `speech` CLI from speech-swift (native Apple-Silicon MLX bundle
`aufklarer/F5TTS-v1-Base-MLX-fp16`). Fallback path: the pip-installable `f5-tts`
package (Apache-2.0) if `speech` is not on PATH.

Long scripts are split into sentences/clauses; each chunk is synthesized
separately and the resulting WAVs are concatenated into one `voice.wav` (mirrors
the old vox `generate_long` behaviour). F5-TTS is happiest with reference clips
<= ~10 s and modest per-call text length, so chunking also improves quality.
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

from .. import config
from ..episode import Episode

# F5 reference clips live in voices/<name>.wav + voices/<name>.txt (transcript).
VOICE_EXTS = (".wav", ".mp3", ".flac", ".ogg")


def _resolve_voice(ep: Episode) -> tuple[Path, Path]:
    """Return (clip, transcript) for the episode's voice name."""
    base = config.VOICES / ep.voice
    clip = None
    for ext in VOICE_EXTS:
        cand = base.with_suffix(ext)
        if cand.exists():
            clip = cand
            break
    if clip is None:
        # try a directory named after the voice containing any audio
        if base.is_dir():
            for f in base.iterdir():
                if f.suffix.lower() in VOICE_EXTS:
                    clip = f
                    break
    if clip is None:
        raise SystemExit(
            f"voice sample missing: {base} (.wav/.mp3/.flac/.ogg or dir/)")

    transcript = base.with_suffix(".txt")
    if not transcript.exists():
        raise SystemExit(
            f"voice transcript missing: {transcript}\n"
            f"F5-TTS needs the EXACT transcript of {clip.name} to clone the voice."
        )
    return clip, transcript


def _chunks(text: str, max_chars: int = 180) -> list[str]:
    """Split into sentence-ish chunks F5 can handle in one call."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) + 1 <= max_chars:
            cur = (cur + " " + p).strip()
        else:
            if cur:
                chunks.append(cur)
            # very long sentence: hard-split on commas/spaces
            if len(p) > max_chars:
                while p:
                    chunks.append(p[:max_chars].strip())
                    p = p[max_chars:]
            else:
                cur = p
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c.strip()]


def _speech_cli(clip: Path, transcript: Path, text: str, out: Path,
                model_dir: Path) -> None:
    """Invoke the speech-swift CLI: speech speak ... --engine f5."""
    model_arg = ["--model", str(model_dir)] if model_dir.exists() else []
    cmd = [
        config.F5_BIN, "speak", text,
        "--engine", "f5",
        "--voice-sample", str(clip),
        "--f5-reference-text", transcript.read_text().strip(),
        "-o", str(out),
        *model_arg,
    ]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)


def _f5_python(clip: Path, transcript: Path, text: str, out: Path) -> None:
    """Fallback: use the pip `f5-tts` package in-process (Apache-2.0).

    Values are inlined into the generated runner script (json-encoded) so the
    child process never depends on argv parsing.
    """
    import json

    ref_text = transcript.read_text().strip()
    runner = (
        "import soundfile as sf\n"
        "from f5_tts.api import F5TTS\n"
        "tts = F5TTS(model='F5TTS_v1_Base')\n"
        "wav, sr, _ = tts.infer(\n"
        f"    ref_file={json.dumps(str(clip))},\n"
        f"    ref_text={json.dumps(ref_text)},\n"
        f"    gen_text={json.dumps(text)},\n"
        "    seed=42,\n"
        ")\n"
        f"sf.write({json.dumps(str(out))}, wav, sr)\n"
    )
    tmp = out.with_suffix(".run.py")
    tmp.write_text(runner)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    subprocess.run(
        [sys.executable, str(tmp)],
        check=True, capture_output=True, text=True, env=env,
    )
    tmp.unlink(missing_ok=True)


def _synth(clip: Path, transcript: Path, text: str, out: Path, model_dir: Path) -> None:
    if shutil.which(config.F5_BIN):
        _speech_cli(clip, transcript, text, out, model_dir)
    else:
        print(f"[tts] '{config.F5_BIN}' not on PATH — using f5-tts python fallback")
        _f5_python(clip, transcript, text, out)


def run(episode_id: str, speed: float = 1.0, voice: str | None = None) -> Episode:
    ep = Episode.load(episode_id)
    if voice:
        ep.voice = voice
        ep.save()
    if not ep.voice:
        ep.voice = config.DEFAULT_VOICE
        ep.save()

    clip, transcript = _resolve_voice(ep)
    text = ep.full_script
    if not text:
        raise SystemExit("episode has no script text to voice")

    chunks = _chunks(text)
    print(f"[tts] voicing {len(chunks)} chunk(s) as '{ep.voice}' "
          f"({len(text.split())} words) …")

    ep.mkdirs()
    tmp_dir = Path(tempfile.mkdtemp(prefix="kc_tts_"))
    wavs: list[Path] = []
    try:
        for i, chunk in enumerate(chunks):
            out = tmp_dir / f"c{i:02d}.wav"
            _synth(clip, transcript, chunk, out, config.F5_MODEL_DIR)
            if not out.exists():
                raise SystemExit(f"TTS produced no audio for chunk {i}: {chunk[:40]!r}")
            wavs.append(out)

        # Concatenate chunks into one voice.wav with ffmpeg (also applies speed).
        dest = ep.audio_dir / "voice.wav"
        concat_list = tmp_dir / "list.txt"
        concat_list.write_text("\n".join(f"file '{w.resolve()}'" for w in wavs))
        cmd = [config.FFMPEG, "-y", "-f", "concat", "-safe", "0",
               "-i", str(concat_list)]
        if speed and speed != 1.0:
            cmd += ["-filter:a", f"atempo={speed}"]
        cmd += ["-c:a", "pcm_s16le", str(dest)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"ffmpeg concat failed:\n{proc.stderr[-1500:]}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    ep.voice_audio = "audio/voice.wav"
    ep.save()
    print(f"[tts] -> {dest}")
    return ep
