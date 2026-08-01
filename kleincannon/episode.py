"""The manifest. Every stage reads it, mutates it, writes it back."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path

from . import config


def slugify(text: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:maxlen].strip("-")


@dataclass
class Beat:
    id: str
    text: str
    image_prompt: str | None = None
    image: str | None = None          # filename inside images/
    audio_start: float | None = None
    audio_end: float | None = None
    motion: str = "in"                # in | out | left | right — Ken Burns direction

    @property
    def duration(self) -> float:
        if self.audio_start is None or self.audio_end is None:
            return 0.0
        return self.audio_end - self.audio_start


@dataclass
class Episode:
    id: str
    topic: str
    hook: str = ""
    purpose: str = ""                 # free-form: what this video is for (no branding logic)
    beats: list[Beat] = field(default_factory=list)
    voice: str = config.DEFAULT_VOICE
    style_suffix: str = ""
    cta: str = config.CTA             # optional free-text CTA (never auto-burned)
    voice_audio: str | None = None    # audio/voice.wav
    words: list[dict] = field(default_factory=list)   # whisper word timestamps
    final: str | None = None

    # ---- paths ----
    @property
    def dir(self) -> Path:
        return config.EPISODES / self.id

    @property
    def images_dir(self) -> Path:
        return self.dir / "images"

    @property
    def audio_dir(self) -> Path:
        return self.dir / "audio"

    @property
    def manifest_path(self) -> Path:
        return self.dir / "episode.json"

    def mkdirs(self) -> None:
        for d in (self.dir, self.images_dir, self.audio_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---- text ----
    @property
    def full_script(self) -> str:
        return " ".join(b.text.strip() for b in self.beats if b.text.strip())

    @property
    def speech_end(self) -> float:
        ends = [b.audio_end for b in self.beats if b.audio_end is not None]
        return max(ends) if ends else 0.0

    @property
    def total_duration(self) -> float:
        return self.speech_end + config.TAIL_SECONDS

    # ---- io ----
    @classmethod
    def new(cls, topic: str, purpose: str = "") -> "Episode":
        eid = f"{date.today().isoformat()}-{slugify(topic)}"
        ep = cls(id=eid, topic=topic, purpose=purpose)
        ep.mkdirs()
        return ep

    @classmethod
    def load(cls, episode_id: str) -> "Episode":
        path = config.EPISODES / episode_id / "episode.json"
        if not path.exists():
            raise FileNotFoundError(f"no manifest at {path}")
        raw = json.loads(path.read_text())
        beats = [Beat(**b) for b in raw.pop("beats", [])]
        return cls(beats=beats, **raw)

    def save(self) -> Path:
        self.mkdirs()
        data = asdict(self)
        self.manifest_path.write_text(json.dumps(data, indent=2))
        return self.manifest_path


def latest_episode_id() -> str:
    eps = sorted(
        (p for p in config.EPISODES.iterdir() if (p / "episode.json").exists()),
        key=lambda p: (p / "episode.json").stat().st_mtime,
    )
    if not eps:
        raise SystemExit("no episodes yet — run the script stage first")
    return eps[-1].name
