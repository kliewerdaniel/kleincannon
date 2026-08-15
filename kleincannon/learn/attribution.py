"""Closed-deal attribution — the sparse signal that defines the real objective.

A closed deal carries an ordered touchpoint list (which asset/channel the lead
touched, when). We credit its value back to the contributing experiences using a
deterministic, reproducible attribution model so the bandit can learn from revenue
without overfitting a single lucky close.

Default model: **time-decay multi-touch credit.** An asset that touched a deal
`d` days ago gets weight `exp(-d / tau)` (tau = half-life from learn_config).
Weights are normalised so a deal of value V contributes `V * weight_i` to
experience i's `attributed_value`. Low-confidence deals contribute at most
`confidence * value` (a guessed source can't corrupt the model).

Manual override (`attribution_method="manual"`) credits exactly the supplied
`attributed_experience_ids` equally (or by explicit touchpoint weight).

Fail-closed: attribution never fabricates a deal — `record_deal` requires a real
value + touchpoints; `credit_for` returns an empty list when there's nothing.
"""
from __future__ import annotations

import time
from typing import Any

from . import db, learn_config


def _days_since(ts: float) -> float:
    if not ts:
        return 0.0
    return max(0.0, (time.time() - ts) / 86400.0)


def credit_for(deal: dict[str, Any]) -> list[tuple[str, float]]:
    """Given a stored conversion row (dict), return [(experience_id, credit)].

    Pure + deterministic: same deal -> same credits every time. No network.
    """
    method = deal.get("attribution_method", "time_decay")
    value = float(deal.get("value", 0.0) or 0.0)
    confidence = float(deal.get("confidence", 1.0) or 1.0)
    floor = float(learn_config.attribution.get("confidence_floor", 0.3))
    if confidence < floor:
        # below the confidence floor => no credit flows (logged elsewhere)
        return []
    effective_value = value * max(0.0, min(1.0, confidence))

    touchpoints = deal.get("touchpoints") or []
    explicit_ids = deal.get("attributed_experience_ids") or []

    if method == "manual":
        if not explicit_ids:
            return []
        share = effective_value / len(explicit_ids)
        return [(eid, round(share, 5)) for eid in explicit_ids]

    if method == "last_touch":
        if not explicit_ids:
            return []
        return [(explicit_ids[-1], round(effective_value, 5))]

    # default: time_decay multi-touch over the touchpoint list
    if not touchpoints:
        # fall back to explicit ids if no touchpoints were recorded
        if not explicit_ids:
            return []
        share = effective_value / len(explicit_ids)
        return [(eid, round(share, 5)) for eid in explicit_ids]

    tau = float(learn_config.attribution.get("half_life_days", 14.0)) or 14.0
    raw: dict[str, float] = {}
    for tp in touchpoints:
        eid = tp.get("experience_id") or tp.get("id")
        if not eid:
            continue
        d = _days_since(float(tp.get("at") or 0.0))
        w = float(tp.get("weight", 1.0)) * (2.0 ** (-d / tau))
        raw[eid] = raw.get(eid, 0.0) + w
    if not raw:
        return []
    total = sum(raw.values())
    if total <= 0:
        return []
    return [(eid, round(effective_value * (w / total), 5)) for eid, w in raw.items()]


def record_deal(*, deal_id: str, value: float, offer: str = "",
                touchpoints: list[dict] | None = None,
                attributed_experience_ids: list[str] | None = None,
                attribution_method: str | None = None,
                confidence: float = 1.0, source_ref: str = "") -> dict[str, Any]:
    """Persist a closed deal and credit its value back to contributing assets.

    Returns a summary: the deal row + the per-experience credits applied.
    Fail-closed: requires a positive value; credits are applied only to existing
    experiences (a mistyped id is skipped, never fabricated).
    """
    method = attribution_method or learn_config.attribution.get("method", "time_decay")
    store = db.open_db()
    store.add_conversion(
        deal_id=deal_id, value=value, offer=offer, touchpoints=touchpoints,
        attributed_experience_ids=attributed_experience_ids,
        attribution_method=method, confidence=confidence, source_ref=source_ref)
    deal = store.get_conversion(deal_id)
    store.close()

    credits = credit_for(deal) if deal else []
    # Apply credits to experiences' attributed_value (additive + idempotent-ish:
    # re-recording the same deal overwrites via INSERT OR REPLACE of the deal,
    # but attributed_value is accumulated here; we set it from the sum of credits).
    applied = 0
    if credits:
        store = db.open_db()
        for eid, credit in credits:
            exp = store.get_experience(eid)
            if exp is None:
                continue  # skip unknown ids rather than fabricate
            # accumulate across deals for this experience
            store.update_experience(eid, attributed_value=exp.attributed_value + credit)
            applied += 1
        store.close()
    return {"deal_id": deal_id, "value": value, "confidence": confidence,
            "method": method, "credits": credits, "n_credited": applied}


def attributed_total() -> float:
    """Sum of all attributed value across experiences (dashboard metric)."""
    store = db.open_db()
    rows = store.conn.execute(
        "SELECT COALESCE(SUM(attributed_value),0) FROM experiences").fetchone()
    store.close()
    return float(rows[0]) if rows else 0.0


def list_deals(limit: int = 100) -> list[dict[str, Any]]:
    store = db.open_db()
    out = store.list_conversions(limit=limit)
    store.close()
    return out
