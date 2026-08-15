"""Knowledge-engine bridge for the Kleincannon content loop.

P1 integration: feed the **Obladaet Knowledge Engine** (the white-label
wrapping of Hermes Atlas) into the content stages so videos about
DanielKliewer.com draw on *compiled, provenanced* knowledge instead of the
model's prior.

Design rules (carry from the kc-unified spec):
  * Fail-closed. If the engine can't be built, isn't compiled, or returns
    nothing, every grounding function returns None / an empty block and the
    caller proceeds UNGROUNDED. We never invent claims or sources.
  * Provenance is sacred. Anything we surface carries its source paths; the
    content stages may quote a claim but must keep the source attached upstream.
  * HITL owns strategy. This module only supplies grounded material; it never
    decides what to say or auto-publishes.

The engine is imported lazily (inside functions) so kleincannon stays runnable
with no obladaet installed — importing this module must never hard-fail.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import config


# Where the compiled Atlas index for the Kleincannon corpus lives. Kept under
# the project, git-ignored (it's a build artifact). Override via env.
KNOWLEDGE_INDEX_DIR = Path(
    os.environ.get("KLEINCANNON_KNOWLEDGE_DIR", str(config.ROOT / "knowledge" / ".atlas"))
)
# The corpus roots fed to the engine. Default: the checked-in knowledge/ folder.
KNOWLEDGE_CORPUS_ROOTS = [
    str(config.ROOT / "knowledge"),
]
# Opt-in flag. The pipeline is generic; grounding only engages when this is set
# (config override / web toggle / env). Mirrors the "strategy set by Daniel"
# HITL boundary — the engine assists, it does not switch itself on.
USE_KNOWLEDGE_ENGINE = os.environ.get("KLEINCANNON_USE_KNOWLEDGE_ENGINE", "").lower() in (
    "1", "true", "yes",
)

_ENGINE = None  # module-level singleton


def _build_engine():
    """Construct a KnowledgeEngine, or None if obladaet is unavailable."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    try:
        from obladaet import KnowledgeEngine  # lazy import — fail-closed
    except Exception:  # noqa: BLE001 — no obladaet installed
        _ENGINE = None
        return None
    _ENGINE = KnowledgeEngine(
        atlas_dir=str(KNOWLEDGE_INDEX_DIR),
        roots=list(KNOWLEDGE_CORPUS_ROOTS),
    )
    return _ENGINE


def is_available() -> bool:
    """True if the engine is installed AND the index is compiled."""
    eng = _build_engine()
    if eng is None:
        return False
    try:
        st = eng.status()
    except Exception:  # noqa: BLE001
        return False
    return bool(st.available and st.compiled)


def ensure_compiled(roots: Optional[list[str]] = None) -> dict:
    """Compile the corpus if needed. Returns a status dict (never raises)."""
    eng = _build_engine()
    if eng is None:
        return {"ok": False, "reason": "obladaet not installed"}
    try:
        res = eng.ingest(roots or KNOWLEDGE_CORPUS_ROOTS)
        return {"ok": bool(res.ok), "sources": res.sources,
                "claims": res.stats.get("claims", 0), "note": res.note}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


@dataclass
class GroundedBlock:
    """Material the content stages can draw on, with provenance attached."""
    query: str
    mode: str                 # "compiled" | "grep" | "empty" | "unavailable"
    degraded: bool
    claims: list[dict[str, Any]] = field(default_factory=list)   # raw ClaimRef dicts
    synthesis: Optional[str] = None
    citations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.claims and not self.synthesis

    def quote_lines(self, max_claims: int = 3) -> list[str]:
        """Short citable one-liners for the script stage (text + source path)."""
        out = []
        for c in self.claims[:max_claims]:
            src = ""
            sources = c.get("sources") or []
            if sources:
                src = sources[0].get("path", "")
            line = c.get("text", "").strip()
            if line:
                out.append({"text": line, "source": src})
        return out


def ground(topic: str, angle: str = "", limit: int = 6) -> GroundedBlock:
    """Retrieve grounded knowledge for a topic/angle.

    Fail-closed: returns an `unavailable`/`empty` block on any error so the
    caller can fall back to ungrounded generation without crashing.
    """
    eng = _build_engine()
    if eng is None:
        return GroundedBlock(query=topic, mode="unavailable", degraded=True)
    try:
        q = topic
        if angle:
            q = f"{topic} {angle}"
        res = eng.query(q, limit=limit)
    except Exception as e:  # noqa: BLE001
        return GroundedBlock(query=topic, mode="unavailable", degraded=True,
                             synthesis=f"[knowledge error: {e}]")
    return GroundedBlock(
        query=q,
        mode=res.mode,
        degraded=res.degraded,
        claims=[c.model_dump() for c in res.claims],
    )


def research(topic: str, angle: str = "", limit: int = 6) -> GroundedBlock:
    """Retrieve + synthesize (no LLM client here — synthesis stays None unless
    the engine was constructed with one). Returns provenance-bearing material."""
    eng = _build_engine()
    if eng is None:
        return GroundedBlock(query=topic, mode="unavailable", degraded=True)
    try:
        q = f"{topic} {angle}".strip() or topic
        res = eng.research(q, limit=limit)
    except Exception as e:  # noqa: BLE001
        return GroundedBlock(query=topic, mode="unavailable", degraded=True,
                             synthesis=f"[knowledge error: {e}]")
    return GroundedBlock(
        query=q,
        mode=res.mode,
        degraded=res.degraded,
        claims=[c.model_dump() for c in res.claims],
        synthesis=res.synthesized,
        citations=[s.model_dump() for s in res.citations],
    )


# ---------------------------------------------------------------------------
# Prompt builders — turn a GroundedBlock into text the LLM stages can paste in.
# These never invent; they only format what the engine returned.
# ---------------------------------------------------------------------------
def format_for_script(block: GroundedBlock) -> str:
    """Evidence block injected into the script-stage system/user prompt.

    Instructs the model to ground the hook/beats in these claims and to KEEP the
    source spirit (no fabrication). Returns '' when empty so callers can skip.
    """
    if block.empty:
        return ""
    lines = []
    if block.degraded:
        lines.append(
            "[Note: knowledge index unavailable — proceeding on model knowledge. "
            "Do NOT present the lines below as fact; they are retrieval hints only.]"
        )
    lines.append("Grounded knowledge to anchor the script (draw real substance "
                 "from these; do not invent specifics beyond them):")
    for i, c in enumerate(block.claims[:4], 1):
        src = ""
        sources = c.get("sources") or []
        if sources:
            src = f" (src: {sources[0].get('title', '') or sources[0].get('path', '')})"
        lines.append(f"  {i}. {c.get('text', '').strip()}{src}")
    return "\n".join(lines)


def format_for_prompts(block: GroundedBlock) -> str:
    """Evidence block for the image-prompt stage — keeps visuals on-topic with
    the actual subject matter rather than generic emotion stock."""
    if block.empty:
        return ""
    lines = ["Grounded subject references (keep imagery tied to these real themes):"]
    for c in block.claims[:3]:
        lines.append(f"  - {c.get('text', '').strip()}")
    return "\n".join(lines)


def grounded_quotes(block: GroundedBlock, max_quotes: int = 3) -> list[str]:
    """Plain citable quote strings for the script stage to weave in."""
    return [q["text"] for q in block.quote_lines(max_quotes)]
