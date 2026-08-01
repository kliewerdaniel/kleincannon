"""Stage 6 — karaoke captions, rasterized with Pillow (no libass on this Mac).

The system ffmpeg has no libass/drawtext filters, so we can't burn ASS
subtitles. Instead we render a *single full-frame transparent caption layer*
as one PNG per video frame, with the spoken word highlighted in CAPTION_ACCENT
and upcoming words in CAPTION_WHITE. assemble.py then composites that one
layer video over the base with a SINGLE ffmpeg `overlay` filter — no fragile
long overlay chains (which ffmpeg silently truncates).

Captions are derived from the AUTHORED beat text (not ASR) so they read
cleanly. Word timings come from ep.words (whisper) aligned to the beats.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .. import config
from ..episode import Episode

WORDS_PER_CARD = 3
MAX_CARD_SECONDS = 1.8

# Use a heavy system font that exists on macOS; the old "Arial Black" ask maps to it.
_FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Supplemental/HelveticaNeue.ttc",
]


def _font(size: int) -> "ImageFont.ImageFont":
    for p in _FONT_PATHS:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _word_timings(ep: Episode) -> list[dict]:
    """Flatten aligned words into a per-word timing list, one per spoken token."""
    out: list[dict] = []
    if ep.words:
        for w in ep.words:
            out.append({"word": w["word"], "start": w["start"], "end": w["end"]})
        return out
    # Fallback: evenly distribute across each beat's audio window.
    for b in ep.beats:
        toks = b.text.split()
        if not toks or b.audio_start is None:
            continue
        span = (b.audio_end or b.audio_start + 1.0) - b.audio_start
        per = span / len(toks)
        for i, t in enumerate(toks):
            out.append({
                "word": t, "start": b.audio_start + i * per,
                "end": b.audio_start + (i + 1) * per,
            })
    return out


def _fit_layout(height: int, lines: list[str], sample_font: ImageFont.FreeTypeFont) -> int:
    """Largest font size (within config fracs) at which every line + the block fit."""
    max_f = int(height * config.CAPTION_FONT_MAX_FRAC)
    min_f = int(height * config.CAPTION_FONT_MIN_FRAC)
    budget = config.WIDTH * config.CAPTION_WIDTH_BUDGET
    max_block = height * config.CAPTION_BLOCK_MAX_FRAC
    best = min_f
    longest = max(lines, key=len) if lines else ""
    for size in range(max_f, min_f - 1, -2):
        f = sample_font.font_variant(size=size)
        line_w = f.getlength(longest) if longest else 0
        ascent, descent = f.getmetrics()
        line_h = ascent + descent
        if line_w <= budget and line_h * len(lines) <= max_block:
            best = size
            break
    return max(best, min_f)


def run(episode_id: str) -> Episode:
    ep = Episode.load(episode_id)
    words = _word_timings(ep)
    if not words:
        raise SystemExit("no words to caption — run tts + align first")

    cap_dir = ep.dir / "captions"
    cap_dir.mkdir(parents=True, exist_ok=True)
    width, height = config.WIDTH, config.HEIGHT

    # Build WORDS_PER_CARD cards from the timed words.
    cards: list[list[dict]] = [words[i:i + WORDS_PER_CARD]
                               for i in range(0, len(words), WORDS_PER_CARD)]

    sample = _font(int(height * config.CAPTION_FONT_MAX_FRAC))
    center_y = int(height * config.CAPTION_TOP_FRAC)

    # Pre-render each card once (transparent PNG with its words on a backing plate).
    card_imgs: list[Image.Image] = []
    card_y: list[int] = []
    for ci, card in enumerate(cards):
        longest = max((w["word"] for w in card), key=len)
        size = _fit_layout(height, [longest], sample)
        font = sample.font_variant(size=size)
        ascent, descent = font.getmetrics()
        line_h = ascent + descent
        card_w = int(max(font.getlength(w["word"]) for w in card))
        pad = int(height * 0.02)
        img = Image.new("RGBA", (card_w + 2 * pad, line_h * len(card) + 2 * pad), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        # semi-transparent backing plate behind the text
        d.rectangle([0, 0, img.width, img.height], fill=config.CAPTION_BACKING)
        y = pad
        for w in card:
            d.text((pad, y), w["word"], font=font, fill=config.CAPTION_WHITE,
                   stroke_width=max(2, size // 24), stroke_fill=config.CAPTION_STROKE)
            y += line_h
        card_imgs.append(img)
        card_y.append(center_y - img.height // 2)

    # Frame-by-frame caption layer: at time t, find the active card and the
    # word within it whose window contains t (highlight it in accent).
    total = ep.total_duration
    n_frames = int(round(total * config.FPS)) + 1
    layer_w, layer_h = width, height

    # index mapping frame -> (card_index, highlight_word_index_or_None)
    def active_for_t(t: float) -> tuple[int, int | None]:
        for ci, card in enumerate(cards):
            c_start = card[0]["start"]
            c_end = card[-1]["end"]
            if c_start - 0.05 <= t <= c_end + 0.12:
                for wi, w in enumerate(card):
                    if w["start"] - 0.04 <= t <= w["end"] + 0.04:
                        return ci, wi
                return ci, None
        return -1, None

    frames_dir = cap_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # clean any stale frames
    for old in frames_dir.glob("frame_*.png"):
        old.unlink()

    sample_f = sample.font_variant(size=int(height * config.CAPTION_FONT_MAX_FRAC))
    for fi in range(n_frames):
        t = fi / config.FPS
        ci, hi = active_for_t(t)
        layer = Image.new("RGBA", (layer_w, layer_h), (0, 0, 0, 0))
        if ci >= 0:
            img = card_imgs[ci]
            x = int((layer_w - img.width) / 2)
            y = card_y[ci]
            if hi is None:
                layer.alpha_composite(img, (x, y))
            else:
                # re-render this card with the highlighted word in accent
                y2 = y
                comp = Image.new("RGBA", (layer_w, layer_h), (0, 0, 0, 0))
                draw = ImageDraw.Draw(comp)
                card = cards[ci]
                longest = max((w["word"] for w in card), key=len)
                size = _fit_layout(height, [longest], sample_f)
                font = sample_f.font_variant(size=size)
                ascent, descent = font.getmetrics()
                line_h = ascent + descent
                pad = int(height * 0.02)
                # plate
                draw.rectangle([x, y, x + img.width, y + img.height],
                               fill=config.CAPTION_BACKING)
                yy = y + pad
                for wi, w in enumerate(card):
                    fill = config.CAPTION_ACCENT if wi == hi else config.CAPTION_WHITE
                    draw.text((x + pad, yy), w["word"], font=font, fill=fill,
                              stroke_width=max(2, size // 24),
                              stroke_fill=config.CAPTION_STROKE)
                    yy += line_h
                layer = comp
        out = frames_dir / f"frame_{fi:05d}.png"
        layer.save(out)

    meta = {
        "frame": [width, height],
        "fps": config.FPS,
        "n_frames": n_frames,
        "frames_dir": str(frames_dir),
        "top_frac": config.CAPTION_TOP_FRAC,
        "words": [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in words],
    }
    (cap_dir / "words.json").write_text(json.dumps(meta, indent=2))
    ep.save()
    print(f"[captions] {n_frames} caption frames -> {frames_dir}/frame_*.png")
    return ep


def _probe_size(ep: Episode) -> tuple[int, int]:
    """Use the first rendered image's real dimensions so captions match it."""
    for b in ep.beats:
        if b.image:
            p = ep.dir / b.image
            if p.exists():
                with Image.open(p) as im:
                    return im.width, im.height
    return config.GEN_WIDTH, config.GEN_HEIGHT
