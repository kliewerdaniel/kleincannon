"""Stage 4 — image prompts, with a style contract so the video doesn't drift.

General-purpose: writes one image prompt per beat for ANY topic. No product- or
brand-specific language. If the LLM is unavailable or slow, a deterministic local
generator takes over so the pipeline never soft-locks.
"""
from __future__ import annotations

import json
import re

from .. import llm, config
from ..episode import Episode


# Note: STYLE_SUFFIX is kept only as a last-resort fallback if the style catalog
# is ever empty. The active look is chosen per-episode from config.STYLE_CATALOG
# (see _resolve_active_style below) so videos stop all looking the same blue.
STYLE_SUFFIX = (
    "cinematic editorial photograph, vertical 9:16 composition, shot on 35mm, "
    "shallow depth of field, moody directional window light, desaturated teal and amber "
    "palette, film grain, high detail, absolutely no text, no writing, no signage, "
    "no readable screens, no numbers, no logos"
)

NEGATIVE = (
    "text, words, letters, typography, captions, subtitles, writing, numbers, digits, "
    "watermark, logo, signature, username, website URL, brand name, labels, "
    "readable screen, UI text, document text, spreadsheet cells, charts, graphs, "
    "book pages, newspaper, magazine text, handwriting, calligraphy, poster text, "
    "blurry, low quality, low resolution, pixelated, artifacts, "
    "cropped, out of frame, deformed hands, extra fingers, mutated, "
    "cartoon, 3d render, illustration"
)

SYSTEM = """You write image-generation prompts for a vertical short-form video (9:16, TikTok / Reels / Shorts) about a specific TOPIC.

Context: these videos are meant to grab and HOLD attention and to feel a little PROVOCATIVE / contrarian — that drives engagement. So the imagery should be emotionally charged and clearly tied to the beat's actual subject, not abstract filler.

Rules:
- You are given a beat's spoken line. Write ONE image prompt per shot that VISUALLY EMBODIES that line's real subject and the feeling behind it. If the beat is about a packed arena, show a packed arena. If it is about a player being ignored for years, show a nearly empty gym with one lonely figure. Make the image UNDENIABLY about the topic, while still being cinematic.
- Be provocative with mood: tension, disbelief, silence-before-the-storm, a crowd erupting, an underdog finally seen. Lean into contrast (empty vs packed, ignored vs celebrated).
- ABSOLUTELY NO TEXT. Never describe anything that would contain readable words, letters, or numbers: no signs, no screens with visible UI, no documents, spreadsheets, charts, graphs, whiteboards, books, papers, labels, posters, jerseys with readable numbers, or phone displays showing text. The image model renders text as garbage. Depict stats/numbers abstractly (a scoreboard glow blurred to nothing, a tidal wave of bodies in a stadium) — never anything legible.
- Vary composition deliberately across ALL shots in the video: alternate wide establishing, medium, close-up detail, over-the-shoulder. Never two identical framings in a row.
- Keep one consistent human subject across the video when a person appears: describe them the same way each time.
- 30-50 words each. Concrete nouns. Cinematic, not adjective soup.

Output STRICT JSON only: {"b01": ["prompt for shot 1", "prompt for shot 2", ...], "b02": [...], ...}
Each beat is a LIST of prompt strings (one per shot)."""

# Phrases that tend to summon rendered text. Stripped from prompts as a last guard.
_TEXT_TRIGGERS = re.compile(
    r"\b(text|words?|letters?|numbers?|digits?|writing|written|caption|subtitle|"
    r"sign(?:age|s)?|label(?:s|led|ed)?|screen|monitor|display|spreadsheet|chart|"
    r"graph|document|paperwork|invoice|receipt|report|book|newspaper|magazine|"
    r"whiteboard|poster|billboard|logo|brand|dashboard|ui|interface|readable|"
    r"legible|typography|font|ticket|lottery|lotto|jackpot|raffle|gamble|casino|"
    r"bet|odds|prize|payout|fortune|millionaire|coin|cash|money|banknote|bank|"
    r"bill|dollar|cent|price|cost|pay|paid|tax|taxes|slip|card|scratch|"
    r"percent|million|billion|thousand|hundred|math|equation|score|statistic|"
    r"data|table|second|seconds|minute|minutes|hour|hours|year|years|day|days|"
    r"win|won|winning|chance|imagine|imagining|hoping|hope)\b",
    re.IGNORECASE,
)

# When a beat is about numbers, money, odds, or time-at-scale, literal prompts
# make FLUX.2-klein render garbled text on tickets/coins/screens/banknotes. We
# depict the feeling abstractly instead — always text-free.
_NUMERIC = re.compile(r"\b\d+(?:[.,]\d+)*\b|%|percent")
_DANGER = re.compile(
    r"\b(lottery|lotto|ticket|jackpot|raffle|gamble|casino|bet|odds|chance|"
    r"win|won|winning|prize|payout|fortune|millionaire|coin|cash|money|banknote|"
    r"bank|bill|dollar|cent|price|cost|pay|paid|tax|taxes|slip|scratch|"
    r"million|billion|thousand|hundred|math|equation|score|statistic|data|"
    r"map|maps|atlas|globe|world|country|countries|united states|continent|"
    r"forecast|prediction|"
    r"second|seconds|minute|minutes|hour|hours|year|years|day|days|"
    r"imagine|imagining|hoping|hope|lost|never|impossible|vain)\b",
    re.IGNORECASE,
)

# Text-free, topic-flavoured scene fragments used by the no-LLM fallback. Each
# beat gets TWO shots (a wide + a detail) so the video stays visually fresh even
# without the LLM. These lean on the beat's EMOTION (not literal nouns) so the
# model never renders a garbled map/number/name. They are deliberately generic
# enough to fit any explainer but concrete enough to feel like a real scene.
_ABSTRACT = [
    # paired (wide, detail) per beat-index slot
    ("a packed arena seen from the upper stands, a sea of raised phones and straining faces under hot house lights",
     "a single sweat-slick face in the crowd, mouth open in shock, caught in the burst of a camera flash"),
    ("a near-empty gym at dusk, one lone figure shooting free throws at a rim, the echo of the ball the only sound",
     "scuffed wooden floor and a toppled empty water bottle, the quiet of a space nobody bothered to fill"),
    ("a young player on the bench, jaw set, watching the game she is not yet allowed into",
     "hands gripping a knee, knuckles white, the tense stillness before being called in"),
    ("a tidal wave of fans pouring through turnstiles, bodies packed shoulder to shoulder, noise as a physical wall",
     "a close hand slapping a barrier, fingers curling through the mesh, desperate to be part of it"),
    ("a single player under a harsh spotlight, the rest of the court swallowed in black, the whole room holding its breath",
     "eyes closed, a slow exhale, the weight of a moment finally arrived after years of waiting"),
    ("two faces a breath apart in argument, one leaning in, the other turning away, the fracture visible",
     "a phone screen glowing in the dark, the only light in a room, a thumbnail frozen mid-play"),
    ("an empty champion's chair on a lit stage, confetti undisturbed on the floor, the silence after the roar",
     "a hand tracing the edge of a trophy, reflection smeared, the question of who really earned it"),
    ("a wide street of a city at night, a crowd spilling out of a bar cheering at a screen, strangers suddenly one",
     "a single figure walking away from the light, head down, the celebration happening to everyone but them"),
]


# Styles the deterministic "auto" pick is allowed to choose. Lifestyle / fashion
# looks ("Vivid Editorial", "Soft Pastel") render as model/editorial shots that
# don't fit a reality-check explainer, so they're reserved for an EXPLICIT
# --style choice (or a bandit recommendation that sets ep.style directly) and
# never auto-selected. The canonical set lives in config.SAFE_AUTO_STYLES; the
# helper below reads it so prompts.py and episode.py never disagree.
def _safe_auto_styles():
    return set(getattr(config, "SAFE_AUTO_STYLES", []))


def _fallback_prompts(ep: Episode, shots_per_beat: int = 2) -> dict[str, list[str]]:
    """Deterministic prompts used when the LLM is unavailable or too slow.

    Returns {beat_id: [shot1_text, shot2_text, ...]}. Each beat gets
    ``shots_per_beat`` scene fragments drawn from ``_ABSTRACT`` so the video
    stays visually fresh. The fragments lean on the beat's EMOTION (not the
    literal nouns) so ZImage Turbo never renders a garbled map/coin/ticket.
    The style suffix (always a photographic look in the auto path) supplies the
    consistent mood.
    """
    angles = [
        "wide establishing shot",
        "medium shot",
        "close-up detail",
        "over-the-shoulder angle",
        "low-angle heroic framing",
        "extreme close-up",
        "wide environmental context",
        "intimate portrait",
    ]
    out: dict[str, list[str]] = {}
    pair_count = len(_ABSTRACT)
    for i, b in enumerate(ep.beats):
        pair = _ABSTRACT[i % pair_count]
        # pair is (wide, detail); we can also pull a second, different pair for
        # extra shots so 3+ shots per beat still vary.
        base_shots = list(pair)
        while len(base_shots) < shots_per_beat:
            other = _ABSTRACT[(i + len(base_shots)) % pair_count]
            base_shots.append(other[1] if len(base_shots) % 2 else other[0])
        chosen = base_shots[:shots_per_beat]
        out[b.id] = [f"{angles[i % len(angles)]}: {s}" for s in chosen]
    return out


def _auto_style_name(episode_id: str) -> str:
    """Name of the catalog style a topic id deterministically maps to.

    The deterministic pick is constrained to SAFE_AUTO_STYLES (photographic /
    cinematic looks that fit a reality-check explainer); lifestyle/fashion
    styles like "Vivid Editorial" are never auto-selected — they're reserved for
    an explicit --style choice.
    """
    from ..episode import style_for_id
    safe = _safe_auto_styles()
    base = style_for_id(episode_id).get("name", "Moody Cinematic")
    if base in safe:
        return base
    # Fall back to the first safe style deterministically (stable per topic).
    return next(iter(safe)) if safe else "Moody Cinematic"


def _resolve_active_style(ep: Episode, style_arg: str) -> dict:
    """Pick the concrete {name, palette, suffix} for this episode.

    Resolution order:
      1. style_arg is a literal catalog name  -> that entry.
      2. style_arg is a custom literal string  -> used verbatim as the suffix
         (style_name recorded as "custom" so the bandit can tell it apart).
      3. style_arg == "auto" (or empty)        -> deterministic from ep.id via the
         catalog (stable per topic, varied across topics). A previously-resolved
         REAL catalog name is kept for re-run consistency; a stale "custom" legacy
         suffix is NOT reused (that would freeze the look and mislabel it).
    """
    from ..episode import resolve_style, style_for_id
    catalog_names = {entry["name"] for entry in config.STYLE_CATALOG}
    arg = (style_arg or "").strip()
    if arg and arg.lower() not in ("auto",):
        return resolve_style(arg)
    # auto: keep a real catalog name resolved on a prior run (consistency);
    # unless it's a lifestyle/fashion look (Vivid Editorial, Soft Pastel) that
    # should never be auto-reused — re-derive a safe cinematic look instead.
    if ep.style_name in catalog_names and ep.style_name in _safe_auto_styles():
        for entry in config.STYLE_CATALOG:
            if entry["name"] == ep.style_name:
                return dict(entry)
    return style_for_id(ep.id)


def run(episode_id: str, style_suffix: str = "", llm_timeout: float = 75.0,
        shots_per_beat: int = 2) -> Episode:
    ep = Episode.load(episode_id)

    # ---- Resolve the active visual style ---------------------------------
    # Priority: explicit suffix arg > config.PROMPT_STYLE override > episode.style.
    # The chosen style is stored on the episode (name + suffix) so the look is
    # reproducible AND fed to the learning engine (so we can learn which styles win).
    chosen = _resolve_active_style(ep, style_suffix or config.PROMPT_STYLE)
    ep.style = chosen["name"] if chosen["name"] not in ("auto",) else ep.style
    ep.style_name = chosen["name"] if chosen["name"] != "auto" else _auto_style_name(ep.id)
    ep.style_suffix = chosen["suffix"] or STYLE_SUFFIX
    ep.save()

    beats_payload = [{"id": b.id, "text": b.text} for b in ep.beats]
    user = (
        f"Video topic: {ep.topic}\n"
        f"Purpose: {ep.purpose}\n\n"
        f"Beats (spoken lines, in order):\n{json.dumps(beats_payload, indent=2)}\n\n"
        f"For EACH beat return a JSON LIST of exactly {shots_per_beat} image prompts "
        f"(one per shot). Return a JSON object mapping each beat id to its list. "
        f"Exactly {len(ep.beats)} keys."
    )

    # P1: ground the image prompts in compiled knowledge when the engine is on
    # (fail-closed — if nothing returns, the model proceeds ungrounded).
    if getattr(config, "USE_KNOWLEDGE_ENGINE", False):
        try:
            from . import knowledge as kb
            block = kb.ground(ep.topic, angle=ep.purpose or "")
            evidence = kb.format_for_prompts(block)
            if evidence:
                user += (
                    f"\n\n--- GROUNDED SUBJECT REFERENCES (keep imagery tied to these "
                    f"real themes) ---\n{evidence}\n---"
                )
        except Exception as e:  # noqa: BLE001
            print(f"[prompts] knowledge grounding skipped ({type(e).__name__})")

    # Try the LLM, but never block the whole pipeline on a flaky model.
    prompts: dict[str, list[str]] | None = None
    if llm.is_up():
        try:
            raw = llm.chat_json(SYSTEM, user, temperature=0.9,
                                timeout=int(llm_timeout))
            # Normalise: allow either a list-of-prompts or a single string per beat.
            prompts = {}
            for bid, val in raw.items():
                if isinstance(val, str):
                    prompts[bid] = [val]
                elif isinstance(val, list):
                    prompts[bid] = [str(v) for v in val if str(v).strip()]
        except Exception as e:  # noqa: BLE001
            print(f"[prompts] LLM failed ({type(e).__name__}); using local fallback")
            prompts = None

    if not prompts:
        prompts = _fallback_prompts(ep, shots_per_beat=shots_per_beat)
        print("[prompts] using deterministic local prompts (no LLM)")

    missing = [b.id for b in ep.beats if not prompts.get(b.id)]
    if missing:
        raise SystemExit(f"model skipped beats: {missing}")

    ep.style_suffix = ep.style_suffix   # already resolved above
    flagged = []
    for b in ep.beats:
        shot_prompts = prompts[b.id][:shots_per_beat] or [prompts[b.id][0]]
        # Strip text-triggering words from each shot as a safety net.
        cleaned = []
        beat_flagged = set()
        for subj in shot_prompts:
            s = subj.strip().rstrip(".")
            for w in sorted(set(m.lower() for m in _TEXT_TRIGGERS.findall(s))):
                beat_flagged.add(w)
                s = re.sub(rf"\b{re.escape(w)}\b", " ", s, flags=re.IGNORECASE)
            s = re.sub(r"\s{2,}", " ", s).strip().rstrip(".")
            cleaned.append(s)
        if beat_flagged:
            flagged.append((b.id, sorted(beat_flagged)))
        b.shots = [f"{c}. {ep.style_suffix}" for c in cleaned]
        # Backward-compat: first shot is the legacy single image_prompt.
        b.image_prompt = b.shots[0]
    ep.save()

    total_shots = sum(len(b.shots) for b in ep.beats)
    print(f"[prompts] {len(ep.beats)} beats -> {total_shots} shot prompts "
          f"({shots_per_beat}/beat)")
    for b in ep.beats:
        for j, s in enumerate(b.shots):
            print(f"  {b.id}.shot{j + 1}  {s[:100]}…")
    if flagged:
        print("\n[prompts] NOTE — text-triggering words were found and STRIPPED from "
              "these beats' prompts (negative prompt adds a second layer of safety):")
        for bid, words in flagged:
            print(f"    {bid}: {', '.join(words)}")
    return ep
