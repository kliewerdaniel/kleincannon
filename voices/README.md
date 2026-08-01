# Reference voices (F5-TTS zero-shot cloning)

F5-TTS is a **zero-shot** voice-cloning model: it needs a short reference clip of
the target voice **plus the exact transcript of that clip**, then it can speak any
text in that voice.

## Layout

For a voice named `chris`:

```
voices/chris.wav   # the reference clip  (3–10s of clean speech, no music)
voices/chris.txt   # the EXACT transcript of chris.wav
```

The `.wav`/`.mp3`/`.flac`/`.ogg` clip is git-ignored (it's a binary asset). Only
the `.txt` transcript is committed. The default voice is `chris` (see
`config.DEFAULT_VOICE`).

> **Safety net:** `tts.py` does NOT trust `chris.txt` blindly. If faster-whisper
> is available it *transcribes the clip itself* and uses that as F5-TTS
> `ref_text`, cached to `chris.reftext`. A hand-written transcript that drifts
> from the audio makes F5-TTS echo the mismatched words into every generated
> line (the "rabbit leak"), so deriving `ref_text` from the clip removes that
> whole class of bug. The committed `.txt` is only a fallback if whisper is
> unavailable.

## Making a transcript

Play the clip and transcribe it **verbatim** — every word, exactly as spoken.
F5-TTS matches the clip's prosody to the transcript; a mismatch makes the clone
sound off. A few sentences is ideal.

Example `chris.txt`:

```
Hey, this is Chris. I run a deal team and I'm always buried in rent rolls.
```

## Adding a new voice

1. Drop `yourname.wav` here.
2. Write `yourname.txt` with its transcript.
3. Run with `--voice yourname` (CLI) or set `voice: yourname` in the web form.

The `chris.wav` shipped in this repo has its transcript **not** committed yet —
author `voices/chris.txt` before running TTS, or point `--voice` at a voice you
have transcribed.
