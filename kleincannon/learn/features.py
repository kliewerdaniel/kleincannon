"""Feature extraction — convert an experience's generation metadata into a
fixed-width numerical feature vector for the learning engine.

The spec asks for four feature families: Content, Video, Posting, Creator.
We compute them with lightweight, dependency-free heuristics (lexical
sentiment/emotion word lists, duration buckets, weekday/hour one-hots). This
keeps the loop runnable offline with zero extra packages. The interface is
stable: if heavier NLP (transformer embeddings) is added later, append to the
vector and bump FEATURE_VERSION — the bandit reads whatever the current
extractor returns.

Function `extract(meta) -> dict[str, float]` returns named features; `vector(meta)`
returns the ordered float list the model consumes.
"""
from __future__ import annotations

from typing import Any

import numpy as np

FEATURE_VERSION = 1

# --- tiny lexicons for content features (no external deps) ---
_POS = {"good", "great", "best", "love", "amazing", "happy", "win", "free",
        "easy", "simple", "secret", "truth", "beautiful", "calm", "smart"}
_NEG = {"bad", "worst", "hate", "hard", "difficult", "wrong", "lose", "lost",
        "never", "fail", "problem", "risk", "danger", "scary", "lie"}
_EMO = {"love", "hate", "amazing", "scary", "excited", "angry", "happy", "sad",
        "surprised", "shocked", "calm", "anxious", "proud", "jealous"}
_HUMOR = {"funny", "lol", "lmao", "joke", "hilarious", "silly", "meme", "laugh"}
_CONTRO = {"shocking", "controversial", "debate", "they", "don't", "won't",
           "lies", "exposed", "cancelled", "vs", "fight", "argue"}
_EDU = {"how", "why", "what", "learn", "guide", "tip", "trick", "science",
        "study", "research", "fact", "explained", "tutorial", "step"}
_CURIO = {"secret", "hidden", "unknown", "mystery", "reveal", "truth", "never",
          "you", "won't", "believe", "wait", "actually", "here's"}
_URGENCY = {"now", "today", "quick", "fast", "before", "hurry", "limited",
            "last", "deadline", "immediately", "stop"}
_HOOK = {"you", "never", "actually", "here's", "wait", "this", "stop", "the",
         "why", "how", "what", "secret", "truth"}
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]


def _tokens(text: str) -> list[str]:
    return [w.strip(".,!?\"'()[]:;").lower() for w in text.replace("\n", " ").split()]


def _count(words: list[str], lex: set[str]) -> int:
    return sum(1 for w in words if w in lex)


def content_features(meta: dict[str, Any]) -> dict[str, float]:
    text = " ".join([
        str(meta.get("hook", "")),
        str(meta.get("script", "")),
        str(meta.get("title", "")),
    ])
    words = _tokens(text)
    n = max(1, len(words))
    sent = (_count(words, _POS) - _count(words, _NEG)) / n
    return {
        "content_sentiment": round(sent, 4),
        "content_emotion": round(_count(words, _EMO) / n, 4),
        "content_humor": round(_count(words, _HUMOR) / n, 4),
        "content_controversy": round(_count(words, _CONTRO) / n, 4),
        "content_educational": round(_count(words, _EDU) / n, 4),
        "content_curiosity": round(_count(words, _CURIO) / n, 4),
        "content_urgency": round(_count(words, _URGENCY) / n, 4),
        # reading level proxy: mean word length (longer = denser)
        "content_reading_level": round(
            sum(len(w) for w in words) / n, 4) if words else 0.0,
    }


def video_features(meta: dict[str, Any]) -> dict[str, float]:
    dur = float(meta.get("video_length", 0.0) or 0.0)
    dur_bucket = min(1.0, dur / 60.0)            # normalise to 60s
    # pacing proxy: words per second
    script = str(meta.get("script", ""))
    wc = max(1, len(_tokens(script)))
    pacing = min(1.0, (wc / dur) / 3.0) if dur else 0.0   # ~3 wps is brisk
    sub_density = 1.0 if str(meta.get("subtitle_style", "")) == "karaoke" else 0.0
    music_cat = 1.0 if str(meta.get("music", "")) not in ("", "none") else 0.0
    return {
        "video_duration_norm": round(dur_bucket, 4),
        "video_pacing": round(pacing, 4),
        "video_cuts": 0.0,                 # single-image Ken Burns => 0 cuts
        "video_subtitle_density": sub_density,
        "video_music_category": music_cat,
    }


def posting_features(meta: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    # weekday one-hot
    day = meta.get("posting_day") or "Unknown"
    for i, wd in enumerate(_WEEKDAYS):
        out[f"post_wd_{wd}"] = 1.0 if day == wd else 0.0
    # hour bucket (0-23 -> 6 buckets of 4h)
    t = meta.get("posting_time") or ""
    hour = -1
    if t and ":" in t:
        try:
            hour = int(str(t).split(":")[0])
        except ValueError:
            hour = -1
    hb = max(0, min(5, hour // 4)) if hour >= 0 else -1
    for i in range(6):
        out[f"post_hr_{i}"] = 1.0 if i == hb else 0.0
    return out


def creator_features(meta: dict[str, Any]) -> dict[str, float]:
    acct = str(meta.get("poster_account", "") or "default")
    niche = str(meta.get("niche", "") or "general")
    # stable hash -> small float so we can carry account/niche identity without
    # exploding the feature space. (A real deployment would one-hot or embed.)
    acct_h = (hash(acct) % 1000) / 1000.0
    niche_h = (hash(niche) % 1000) / 1000.0
    return {
        "creator_account_id": round(acct_h, 4),
        "creator_niche_id": round(niche_h, 4),
    }


def extract(meta: dict[str, Any]) -> dict[str, float]:
    feats = {}
    feats.update(content_features(meta))
    feats.update(video_features(meta))
    feats.update(posting_features(meta))
    feats.update(creator_features(meta))
    return feats


def vector(meta: dict[str, Any]) -> np.ndarray:
    feats = extract(meta)
    # stable ordering
    keys = sorted(feats.keys())
    return np.array([feats[k] for k in keys], dtype=float)


def feature_names() -> list[str]:
    # a representative metadata sample to enumerate keys in stable order
    sample = {
        "hook": "", "script": "", "title": "", "video_length": 0, "subtitle_style": "",
        "music": "", "posting_day": "Unknown", "posting_time": "", "poster_account": "",
        "niche": "", "topic": "",
    }
    return sorted(extract(sample).keys())


def feature_dim() -> int:
    return len(feature_names())
