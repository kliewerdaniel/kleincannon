"""Stage 6 — karaoke captions, rasterized with Pillow (no libass on this Mac).

The system ffmpeg has no libass/drawtext filters, so we can't burn ASS subtitles.
Instead we rasterize each *word* to a transparent PNG exactly when it is spoken,
then composite them in assemble.py with the ffmpeg `overlay` filter (which IS
available). Spoken words render in CAPTION_ACCENT; upcoming words in CAPTION_WHITE.

Captions are derived from the AUTHORED beat text (not ASR) so they read cleanly.
Word timings come from ep.words (whisper) aligned to the beats.
"""
from __future__ import annotations

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


def _fit_layout(width: int, height: int, lines: list[str], sample_font: ImageFont.FreeTypeFont) -> int:
    """Largest font size (within config fracs) at which every line + the block fit."""
    max_f = int(height * config.CAPTION_FONT_MAX_FRAC)
    min_f = int(height * config.CAPTION_FONT_MIN_FRAC)
    budget = width * config.CAPTION_WIDTH_BUDGET
    max_block = height * config.CAPTION_BLOCK_MAX_FRAC
    best = min_f
    # find the longest line to size against
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


def _rasterize(words: list[dict], width: int, height: int, out_dir: Path) -> list[dict]:
    """Render one transparent PNG per spoken-word card. Returns a timing table
    used by assemble.py (index -> {png, start, end, x, y})."""
    cards: list[list[dict]] = []
    for i in range(0, len(words), WORDS_PER_CARD):
        cards.append(words[i:i + WORDS_PER_CARD])

    sample = _font(int(height * config.CAPTION_FONT_MAX_FRAC))
    table: list[dict] = []
    center_y = int(height * config.CAPTION_TOP_FRAC)

    for ci, card in enumerate(cards):
        longest = max((w["word"] for w in card), key=len)
        size = _fit_layout(width, height, [longest], sample)
        font = sample.font_variant(size=size)
        ascent, descent = font.getmetrics()
        line_h = ascent + descent

        card_w = int(max(font.getlength(w["word"]) for w in card))
        pad = int(height * 0.02)
        img = Image.new("RGBA", (card_w + 2 * pad, line_h + 2 * pad), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        y = pad
        idx_start = len(table)
        for w in card:
            d.text((pad, y), w["word"], font=font, fill=config.CAPTION_WHITE,
                   stroke_width=max(2, size // 24), stroke_fill=config.CAPTION_STROKE)
            table.append({"word": w["word"], "start": w["start"], "end": w["end"]})
            y += line_h
        png = out_dir / f"card_{ci:03d}.png"
        img.save(png)
        # placement (top-left of the card, centered horizontally, near lower third)
        x = int((width - img.width) / 2)
        y_top = center_y - img.height // 2
        for j in range(len(card)):
            table[idx_start + j]["png"] = str(png)
            table[idx_start + j]["x"] = x
            table[idx_start + j]["y"] = y_top
    return table


def run(episode_id: str) -> Episode:
    ep = Episode.load(episode_id)
    words = _word_timings(ep)
    if not words:
        raise SystemExit("no words to caption — run tts + align first")

    cap_dir = ep.dir / "captions"
    cap_dir.mkdir(parents=True, exist_ok=True)
    width, height = _probe_size(ep)
    table = _rasterize(words, width, height, cap_dir)

    meta = {
        "frame": [width, height],
        "top_frac": config.CAPTION_TOP_FRAC,
        "words": table,
    }
    (cap_dir / "words.json").write_text(__import__("json").dumps(meta, indent=2))
    ep.save()
    print(f"[captions] {len(table)} word cards rasterized -> {cap_dir}/words.json")
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
