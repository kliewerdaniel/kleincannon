"""Stage 6 — karaoke captions, rasterized with Pillow (no libass on this Mac).

The system ffmpeg has no libass/drawtext filters, so we can't burn ASS
subtitles. Instead we render a *single full-frame transparent caption layer*
as one PNG per video frame, with the spoken word highlighted in CAPTION_ACCENT
and the rest of the card in CAPTION_WHITE. assemble.py then composites that one
layer over the base with a SINGLE ffmpeg `overlay` filter — no fragile long
overlay chains (which ffmpeg silently truncates).

Layout rule: a card holds several words laid out as WORDS ON A LINE, wrapped
into at most CAPTION_MAX_LINES lines, with the font size and the per-word x/y
positions computed ONCE per card. Both the idle render and the highlighted
render use that same geometry, so words can never land on top of each other.

Captions are derived from whisper word timings aligned to the authored beats.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .. import config
from ..episode import Episode

# Use a heavy system font that exists on macOS; the old "Arial Black" ask maps to it.
_FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Supplemental/HelveticaNeue.ttc",
]


def _font(size: int) -> "ImageFont.FreeTypeFont":
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


def _group_cards(words: list[dict]) -> list[list[dict]]:
    """Chunk timed words into cards of <= CAPTION_WORDS_PER_CARD words.

    A card is also closed early when the gap to the next word is long (a
    natural pause) or when the card has already been on screen for
    CAPTION_MAX_CARD_SECONDS, so the text keeps pace with the speech.
    """
    per_card = config.CAPTION_WORDS_PER_CARD
    max_s = config.CAPTION_MAX_CARD_SECONDS
    cards: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        if cur:
            span = w["end"] - cur[0]["start"]
            gap = w["start"] - cur[-1]["end"]
            if len(cur) >= per_card or span > max_s or gap > 0.7:
                cards.append(cur)
                cur = []
        cur.append(w)
    if cur:
        cards.append(cur)
    return cards


def _layout_card(card: list[dict], font_cache: dict[int, ImageFont.FreeTypeFont]) -> dict:
    """Pick the largest font at which the card wraps into <= max lines and fits.

    Returns geometry: font size, line height, plate box, and one (x, y) per word
    — computed ONCE so idle and highlighted renders are pixel-identical.
    """
    height, width = config.HEIGHT, config.WIDTH
    budget = width * config.CAPTION_WIDTH_BUDGET
    max_block = height * config.CAPTION_BLOCK_MAX_FRAC
    max_f = int(height * config.CAPTION_FONT_MAX_FRAC)
    min_f = int(height * config.CAPTION_FONT_MIN_FRAC)
    tokens = [w["word"] for w in card]

    chosen = None
    for size in range(max_f, min_f - 1, -2):
        f = font_cache.get(size)
        if f is None:
            f = font_cache[size] = _font(size)
        space = f.getlength(" ")
        widths = [f.getlength(t) for t in tokens]
        if max(widths) > budget:
            continue
        # greedy wrap
        lines: list[list[int]] = [[]]
        line_w = 0.0
        for i, wpx in enumerate(widths):
            add = wpx if not lines[-1] else space + wpx
            if lines[-1] and line_w + add > budget:
                lines.append([i])
                line_w = wpx
            else:
                lines[-1].append(i)
                line_w += add
        ascent, descent = f.getmetrics()
        line_h = int((ascent + descent) * 1.08)
        if len(lines) <= config.CAPTION_MAX_LINES and line_h * len(lines) <= max_block:
            chosen = (size, f, lines, widths, space, line_h)
            break

    if chosen is None:
        # floor: use the min font and accept whatever wrap it produces
        size = min_f
        f = font_cache.setdefault(size, _font(size))
        space = f.getlength(" ")
        widths = [f.getlength(t) for t in tokens]
        lines = [[]]
        line_w = 0.0
        for i, wpx in enumerate(widths):
            add = wpx if not lines[-1] else space + wpx
            if lines[-1] and line_w + add > budget:
                lines.append([i])
                line_w = wpx
            else:
                lines[-1].append(i)
                line_w += add
        ascent, descent = f.getmetrics()
        line_h = int((ascent + descent) * 1.08)
        chosen = (size, f, lines, widths, space, line_h)

    size, f, lines, widths, space, line_h = chosen
    pad_x = int(size * 0.55)
    pad_y = int(size * 0.35)

    line_widths = [
        sum(widths[i] for i in ln) + space * max(0, len(ln) - 1) for ln in lines
    ]
    block_w = int(max(line_widths)) if line_widths else 1
    block_h = line_h * len(lines)
    plate_w = block_w + 2 * pad_x
    plate_h = block_h + 2 * pad_y

    plate_x = int((config.WIDTH - plate_w) / 2)
    plate_y = int(config.HEIGHT * config.CAPTION_TOP_FRAC - plate_h / 2)

    positions: list[tuple[int, int]] = [(0, 0)] * len(card)
    for li, ln in enumerate(lines):
        x = plate_x + pad_x + int((block_w - line_widths[li]) / 2)
        y = plate_y + pad_y + li * line_h
        for i in ln:
            positions[i] = (int(x), int(y))
            x += widths[i] + space

    return {
        "size": size, "font": f, "line_h": line_h,
        "plate": (plate_x, plate_y, plate_x + plate_w, plate_y + plate_h),
        "positions": positions,
    }


def _draw_card(card: list[dict], geo: dict, highlight: int | None) -> Image.Image:
    layer = Image.new("RGBA", (config.WIDTH, config.HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x0, y0, x1, y1 = geo["plate"]
    r = max(8, geo["size"] // 4)
    d.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=config.CAPTION_BACKING)
    font = geo["font"]
    stroke = max(2, geo["size"] // 20)
    for i, w in enumerate(card):
        fill = config.CAPTION_ACCENT if i == highlight else config.CAPTION_WHITE
        d.text(geo["positions"][i], w["word"], font=font, fill=fill,
               stroke_width=stroke, stroke_fill=config.CAPTION_STROKE)
    return layer


def run(episode_id: str) -> Episode:
    ep = Episode.load(episode_id)
    # The manifest's word timings must match the current voice.wav. If TTS was
    # re-run since the last align, the saved timings are stale (captions would
    # desync from the speech) — realign before captioning.
    import kleincannon.stages.align as align_stage
    if align_stage.needs_realign(ep):
        print("[captions] voice.wav newer than manifest — realigning first …")
        align_stage.run(episode_id)
        ep = Episode.load(episode_id)
    words = _word_timings(ep)
    if not words:
        raise SystemExit("no words to caption — run tts + align first")

    cap_dir = ep.dir / "captions"
    cap_dir.mkdir(parents=True, exist_ok=True)

    cards = _group_cards(words)
    font_cache: dict[int, ImageFont.FreeTypeFont] = {}
    geos = [_layout_card(c, font_cache) for c in cards]

    total = ep.total_duration
    n_frames = int(round(total * config.FPS)) + 1

    def active_for_t(t: float) -> tuple[int, int | None]:
        for ci, card in enumerate(cards):
            c_start = card[0]["start"]
            c_end = card[-1]["end"]
            if ci == len(cards) - 1:
                c_end = max(c_end, total)   # last card covers through the end
            if c_start - 0.08 <= t <= c_end + 0.15:
                for wi, w in enumerate(card):
                    w_end = w["end"] if ci != len(cards) - 1 else max(w["end"], total)
                    if w["start"] - 0.04 <= t <= w_end + 0.04:
                        return ci, wi
                return ci, None
        return -1, None

    frames_dir = cap_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("frame_*.png"):
        old.unlink()

    # Each card has at most len(card)+1 distinct renders (idle + one per word) —
    # render them once and reuse across frames instead of redrawing every frame.
    cache: dict[tuple[int, int | None], Image.Image] = {}
    blank = Image.new("RGBA", (config.WIDTH, config.HEIGHT), (0, 0, 0, 0))

    for fi in range(n_frames):
        t = fi / config.FPS
        key = active_for_t(t)
        ci, hi = key
        if ci < 0:
            img = blank
        else:
            img = cache.get(key)
            if img is None:
                img = cache[key] = _draw_card(cards[ci], geos[ci], hi)
        img.save(frames_dir / f"frame_{fi:05d}.png")

    meta = {
        "frame": [config.WIDTH, config.HEIGHT],
        "fps": config.FPS,
        "n_frames": n_frames,
        "frames_dir": str(frames_dir),
        "top_frac": config.CAPTION_TOP_FRAC,
        "words_per_card": config.CAPTION_WORDS_PER_CARD,
        "words": [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in words],
    }
    (cap_dir / "words.json").write_text(json.dumps(meta, indent=2))
    ep.save()
    print(f"[captions] {len(cards)} cards, {n_frames} frames -> {frames_dir}/frame_*.png")
    return ep
