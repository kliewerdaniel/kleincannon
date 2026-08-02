"""Stage 3 — word-level alignment. Removes manual timeline sync.

Transcribes the generated voice track with faster-whisper, then maps the words
back onto the authored beats so every beat gets exact audio_start/audio_end.
We align to the AUTHORED text (not the transcript) so captions read correctly
even when the ASR mishears a word.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from .. import config
from ..episode import Episode

MODEL_SIZE = "small.en"


def _norm(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower())


def transcribe(wav_path: str) -> list[dict]:
    from faster_whisper import WhisperModel

    model = WhisperModel(config.ALIGN_MODEL, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(wav_path, word_timestamps=True, vad_filter=False)
    words = []
    for seg in segments:
        for w in seg.words or []:
            token = _norm(w.word)
            if token:
                words.append({"word": w.word.strip(), "key": token,
                              "start": float(w.start), "end": float(w.end)})
    return words


def map_beats(ep: Episode, words: list[dict]) -> None:
    """Align authored beat words to ASR words via a single global diff."""
    script_words: list[tuple[int, str]] = []
    for bi, beat in enumerate(ep.beats):
        for tok in beat.text.split():
            key = _norm(tok)
            if key:
                script_words.append((bi, key))

    a = [k for _, k in script_words]
    b = [w["key"] for w in words]
    matcher = SequenceMatcher(a=a, b=b, autojunk=False)

    hits: dict[int, list[dict]] = {}
    for i, j, n in matcher.get_matching_blocks():
        for off in range(n):
            beat_index = script_words[i + off][0]
            hits.setdefault(beat_index, []).append(words[j + off])

    prev_end = 0.0
    for bi, beat in enumerate(ep.beats):
        matched = hits.get(bi)
        if matched:
            beat.audio_start = round(min(m["start"] for m in matched), 3)
            beat.audio_end = round(max(m["end"] for m in matched), 3)
        else:
            beat.audio_start = round(prev_end, 3)
            beat.audio_end = round(prev_end + max(1.0, len(beat.text.split()) / 2.6), 3)
        beat.audio_start = round(max(beat.audio_start, prev_end), 3)
        if beat.audio_end <= beat.audio_start:
            beat.audio_end = round(beat.audio_start + 0.8, 3)
        prev_end = beat.audio_end

    for i in range(len(ep.beats) - 1):
        ep.beats[i].audio_end = ep.beats[i + 1].audio_start


def needs_realign(ep: Episode) -> bool:
    """True when the voice audio is newer than the saved manifest.

    That means the manifest's word timings are stale and `align` must be
    re-run before `captions`/`build` — otherwise captions desync from the
    speech and the build truncates the narration to the old duration.

    This is what bit us: after the TTS ref-text fix regenerated `voice.wav`
    nobody re-ran `align`, so the manifest kept the old timings. Stages that
    consume the manifest now call this and auto-realign instead of using
    stale data.
    """
    wav = ep.dir / ep.voice_audio if ep.voice_audio else None
    if not wav or not wav.exists():
        return False
    if not ep.manifest_path.exists():
        return True
    # +1s grace so a re-save that lands microseconds after the wav doesn't
    # flip the flag (save happens right after synthesis in tts.run).
    return wav.stat().st_mtime > ep.manifest_path.stat().st_mtime + 1.0


def needs_recaption(ep: Episode) -> bool:
    """True when the rendered caption frames are older than the data they
    depend on (the aligned manifest and the voice audio).

    If either has changed since the caption frames were rendered, the
    karaoke overlay is stale: it desyncs from the speech and the highlight
    runs past (or short of) the audio so the last words are never shown. This
    is what produced the "captions cut off at the end" regression — a `build`
    ran against caption frames rendered for a *different* TTS run. `build`
    now calls this and auto-re-renders the captions instead of using stale
    frames.
    """
    frames_dir = ep.dir / "captions" / "frames"
    if not frames_dir.exists():
        return True
    frames = list(frames_dir.glob("frame_*.png"))
    if not frames:
        return True
    newest_frame = max(f.stat().st_mtime for f in frames)
    ref = 0.0
    if ep.manifest_path.exists():
        ref = max(ref, ep.manifest_path.stat().st_mtime)
    wav = ep.dir / ep.voice_audio if ep.voice_audio else None
    if wav and wav.exists():
        ref = max(ref, wav.stat().st_mtime)
    # frames must be newer than the data they are derived from
    return newest_frame < ref - 0.5


def run(episode_id: str) -> Episode:
    ep = Episode.load(episode_id)
    if not ep.voice_audio:
        raise SystemExit("run the tts stage first")
    wav = ep.dir / ep.voice_audio
    if not wav.exists():
        raise SystemExit(f"missing {wav}")

    print(f"[align] transcribing {wav.name} …")
    words = transcribe(str(wav))
    if not words:
        raise SystemExit("whisper returned no words")

    ep.words = words
    map_beats(ep, words)
    ep.save()

    print(f"[align] {len(words)} words, speech ends at {ep.speech_end:.2f}s")
    for b in ep.beats:
        print(f"  {b.id}  {b.audio_start:>6.2f} → {b.audio_end:>6.2f}  ({b.duration:.2f}s)")
    return ep
