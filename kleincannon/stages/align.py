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
