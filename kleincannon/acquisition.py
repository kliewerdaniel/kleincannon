"""DanielKliewer.com acquisition context + HITL copy guardrails.

P1 integration: makes the Kleincannon loop *DanielKliewer.com-aware* without
hard-wiring product logic into the generic pipeline. The acquisition context is
a preset the run form / CLI can apply; the copy guardrail is a fail-open HITL
safety net that FLAGS risky copy for Daniel's review — it never silently edits
or blocks (strategy stays with the operator).

Aligned with the kc-unified locked decisions:
  * Target: DanielKliewer.com — sells local-first AI systems / agent products /
    the white-label Obladaet engine.
  * ICP: builders, founders, agencies, technical operators buying local-first
    AI systems / the white-label engine itself.
  * HITL boundary: strategy (offer/ICP/copy guardrails) set by Daniel; engine
    runs execution and flags anomalies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Acquisition context preset
# ---------------------------------------------------------------------------
@dataclass
class AcquisitionContext:
    brand: str = "DanielKliewer.com"
    purpose: str = (
        "Acquire technical operators (builders, founders, agencies) into "
        "Daniel Kliewer's local-first AI systems and the white-label Obladaet "
        "Knowledge Engine."
    )
    # Soft CTA surfaced to the operator (never auto-burned into the video per
    # the no-auto-burn rule; may appear in the post caption).
    cta: str = "Explore the local-first stack at DanielKliewer.com"
    # Brand names the engine is allowed to mention. Anything outside this set in
    # generated copy is flagged (we don't want the engine leaking competitor or
    # unrelated brand names).
    allowed_brands: list[str] = field(default_factory=lambda: [
        "DanielKliewer.com", "Daniel Kliewer", "Obladaet", "Hermes Atlas",
        "Kleincannon", "SovereignSpec", "SovereignRecipe",
    ])
    # Terms that are off-guardrail if asserted as fact (the engine must not
    # claim these without Daniel's sign-off). Heuristic, not legal advice.
    forbidden_claims: list[str] = field(default_factory=lambda: [
        r"\bguarantee[d]?\b.*\b(revenue|deal|customer|sale)\b",
        r"\b100%\b.*\b(success|win|uptime|private|secure)\b",
        r"\bno\b.*\bthird[- ]?party\b.*\b(ev[er]+|always)\b",
        r"\breplace[sd]?\b.*\byour\b.*\bteam\b",
    ])
    # Words that trip a softer review flag (sensitive but not necessarily wrong).
    sensitive_terms: list[str] = field(default_factory=lambda: [
        "sovereign", "sovereignty", "license", "commercial", "resell",
        "white-label", "white label", "money-back", "free", "open-source",
        "open core", "MIT",
    ])


# Module-level default context. The web form / CLI can pass an override.
DKC_PRESET = AcquisitionContext()


# ---------------------------------------------------------------------------
# Copy guardrail (HITL: flag, never block)
# ---------------------------------------------------------------------------
@dataclass
class CopyFlags:
    ok: bool = True                      # True = nothing flagged
    brand_leak: list[str] = field(default_factory=list)     # disallowed brands found
    forbidden_claims: list[str] = field(default_factory=list)
    sensitive_terms: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "brand_leak": self.brand_leak,
            "forbidden_claims": self.forbidden_claims,
            "sensitive_terms": self.sensitive_terms,
            "notes": self.notes,
        }


# A broad brand-word list to *detect* leakage (competitors / unrelated names we
# should not be emitting). Kept conservative; anything not in allowed_brands and
# matching a known brand token is flagged for review.
_BRAND_TOKEN = re.compile(
    r"\b([A-Z][A-Za-z0-9]+(?:[.\-][A-Z][A-Za-z0-9]+)*)\b"
)


def check_copy(text: str, ctx: AcquisitionContext | None = None) -> CopyFlags:
    """Scan generated copy for HITL issues. Fail-open: on any error, return ok=True
    with a note rather than blocking the pipeline."""
    ctx = ctx or DKC_PRESET
    flags = CopyFlags()
    if not text or not text.strip():
        flags.notes.append("empty copy — nothing to check")
        return flags
    try:
        lowered = text.lower()

        # 1. Forbidden claims (assertions we must not make without sign-off).
        for pat in ctx.forbidden_claims:
            if re.search(pat, text, flags=re.IGNORECASE):
                flags.forbidden_claims.append(pat)
        if flags.forbidden_claims:
            flags.ok = False

        # 2. Sensitive terms — flag for review, not a hard fail.
        for term in ctx.sensitive_terms:
            if term.lower() in lowered:
                flags.sensitive_terms.append(term)

        # 3. Brand leakage: any Capitalized brand-like token not in allowed set.
        allowed_lower = {b.lower() for b in ctx.allowed_brands}
        for m in _BRAND_TOKEN.findall(text):
            # skip single common words and our own allowed brands
            if m.lower() in allowed_lower:
                continue
            if m.lower() in {"i", "the", "a", "an", "we", "you", "it", "this",
                             "that", "our", "your", "daniel", "kliewer"}:
                continue
            # Only flag multi-word or clearly brand-shaped tokens (contain a dot
            # or dash, or are >= 4 chars and not a common sentence starter).
            if ("." in m or "-" in m or len(m) >= 4) and m not in flags.brand_leak:
                # Heuristic: ignore ordinary capitalized words unless they look
                # brand-shaped (dot/dash) to avoid false positives.
                if "." in m or "-" in m:
                    flags.brand_leak.append(m)
        if flags.brand_leak:
            flags.ok = False
    except Exception as e:  # noqa: BLE001 — fail-open
        flags.notes.append(f"check_copy error (skipped): {e}")
    return flags


def acquire_purpose(preset: AcquisitionContext | None = None) -> str:
    """Return the DanielKliewer.com acquisition purpose string for the script
    stage's `purpose` field."""
    return (preset or DKC_PRESET).purpose


def acquire_cta(preset: AcquisitionContext | None = None) -> str:
    return (preset or DKC_PRESET).cta
