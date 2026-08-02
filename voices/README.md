# Reference voices (Qwen3-TTS in-context voice cloning)

Qwen3-TTS is an **in-context-learning** voice-cloning model: give it a short
reference clip of the target voice and it speaks any text in that voice. Unlike
F5-TTS, it does **not** need a hand-written transcript — `tts.py` transcribes
each clip automatically with faster-whisper and uses that as `ref_text`.

## Layout

For a voice named `chris`:

```
voices/chris.wav   # the reference clip  (5–30s of clean speech, no music)
```

Only the `.wav`/`.mp3`/`.flac`/`.m4a`/`.ogg` clip is needed. It is git-ignored
(binary asset). The auto-transcript is cached to `chris.reftext` (also
ignored). The default voice is `chris` (`config.DEFAULT_VOICE`).

> **No manual transcript required.** Drop a clip, run `kc tts --voice chris`,
> and the engine clones it. If you want to verify/override the transcription,
> inspect or edit `voices/<name>.reftext` (it is regenerated only if the clip
> changes).

## Available voices

The full `chris.wav` (the original, uncut clip) ships by default. The
`vox` companion app's custom voices are also available here:

```
A.wav  John.mp3  Me.wav  chris.wav  elon.mp3  joe.mp3
obama.mp3  rasPutin.mp3  ryan.mp3  ryan.wav  trump.mp3  zuck.mp3
```

Pick any with `--voice <name>` (CLI) or `voice: <name>` (web form), e.g.:

```bash
./kc tts -e 2026-08-01-why-the-ocean-is-salty --voice ryan
./kc all --topic "..." --voice obama
```

## Adding a new voice

1. Drop `yourname.wav` (or `.mp3`/`.flac`/`.m4a`/`.ogg`) here.
2. Run with `--voice yourname`. The transcript is derived automatically.

Best results come from 5–30s of clean, single-speaker speech with minimal
background noise and no music.
