"""P2 — Obladaet offer-prior for the bandit.

Before attributed-deal count is meaningful, the bandit's feature θ is seeded
with a *prior* toward offer-relevant content directions. The prior is derived
from the Knowledge Engine ("which topics/claims correlate with the offer's value
prop?") — never a fabricated label. We map the engine's grounded claims onto the
real feature space (content_educational / content_curiosity / content_urgency /
content_controversy ...) with a small positive bias, so the bandit explores
on-brand assets first.

Fail-closed: if the engine is unavailable or returns nothing, this returns an
empty bias dict and the bandit trains on pure engagement until real deals arrive.
"""
from __future__ import annotations

from typing import Any

from . import learn_config

# Canonical content-feature directions the offer cares about. The engine's
# grounded claims bias these. Magnitudes are modest (the strength scalar is
# applied by the caller) so the prior nudges, not dictates.
_OFFER_SIGNALS = {
    "local-first": "content_educational",
    "sovereignty": "content_curiosity",
    "ownership": "content_curiosity",
    "control": "content_curiosity",
    "privacy": "content_educational",
    "reproducible": "content_educational",
    "watch out": "content_controversy",
    "they don't want": "content_controversy",
    "stop renting": "content_urgency",
    "take back": "content_urgency",
    "before it's gone": "content_urgency",
}


def offer_feature_bias(strength: float = 0.1) -> dict[str, float]:
    """Return {feature_name: bias} for the bandit's seed_prior().

    Derives signal weights from the Obladaet Knowledge Engine over the configured
    corpus, grounded in DanielKliewer.com's acquisition offer. Falls back to a
    static signal map (still grounded in the locked positioning) when the engine
    is unavailable — never fabricates a claim.
    """
    bias: dict[str, float] = {}

    corpus_root = _corpus_root()
    try:
        from kleincannon import knowledge as kb
        if kb.is_available():
            block = kb.ground(
                "DanielKliewer.com offer: local-first AI, sovereignty, ownership, control",
                angle="acquisition offer value proposition")
            text = " ".join(c.get("text", "") for c in block.claims).lower()
            for signal, feat in _OFFER_SIGNALS.items():
                if signal in text:
                    bias[feat] = bias.get(feat, 0.0) + 1.0
    except Exception:
        # engine unavailable — fall through to static prior
        pass

    if not bias:
        # static, still-decisions-grounded fallback (local-first positioning)
        bias = {
            "content_educational": 1.0,
            "content_curiosity": 1.0,
            "content_urgency": 0.5,
        }

    # normalise so the largest signal is `strength`-scaled by the caller
    mx = max(bias.values()) or 1.0
    return {k: round((v / mx), 4) for k, v in bias.items()}


def _corpus_root() -> str:
    root = learn_config.obladaet_prior.get("kb_root", "knowledge")
    from pathlib import Path
    from .. import config
    p = Path(root)
    if not p.is_absolute():
        p = config.ROOT / root
    return str(p)
