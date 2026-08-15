"""P1 integration tests — run against the REAL Obladaet engine + corpus.

These do not mock the knowledge engine. They compile the checked-in
`knowledge/` corpus through the actual vendored Hermes Atlas and assert that
grounding retrieves provenanced claims. A fail-closed test asserts the bridge
degrades gracefully when the engine is unavailable.

Run from the kleincannon repo:
    env -u PYTHONPATH -u PYTHONHOME venv/bin/python -m pytest kleincannon/tests/ -q
"""
from __future__ import annotations

import os

import pytest

from kleincannon import knowledge as kb
from kleincannon import acquisition as acq
from kleincannon import config


HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
CORPUS = os.path.join(REPO_ROOT, "knowledge")


def _ensure_compiled():
    res = kb.ensure_compiled([CORPUS])
    return res


@pytest.fixture(scope="module", autouse=True)
def compiled_index():
    """Compile the corpus once for the module (real Atlas)."""
    return _ensure_compiled()


# ---------------------------------------------------------------------------
# Knowledge engine — real grounding
# ---------------------------------------------------------------------------
def test_engine_installs_and_compiles(compiled_index):
    assert kb._build_engine() is not None, "obladaet should be installed in venv"
    assert compiled_index["ok"] is True, f"compile failed: {compiled_index}"
    assert compiled_index["sources"] >= 1


def test_is_available_true_after_compile():
    assert kb.is_available() is True


def test_ground_returns_provenanced_claims():
    block = kb.ground("local-first AI systems", angle="sovereignty")
    assert block.mode == "compiled", block.mode
    assert block.degraded is False
    assert len(block.claims) >= 1
    # Provenance is sacred: every compiled claim must carry a source with a path.
    for c in block.claims:
        sources = c.get("sources") or []
        assert sources, f"claim {c.get('claim_id')} has no sources"
        assert sources[0].get("path"), "source path is empty"


def test_format_for_script_carries_claims():
    block = kb.ground("Obladaet white-label engine")
    text = kb.format_for_script(block)
    assert text, "format_for_script returned empty for a non-empty block"
    assert "Grounded knowledge" in text


def test_format_for_prompts_carries_themes():
    block = kb.ground("who is the ideal customer for local-first AI")
    text = kb.format_for_prompts(block)
    assert text and "Grounded subject references" in text


def test_grounded_quotes_nonempty():
    block = kb.ground("DanielKliewer.com acquisition")
    quotes = kb.grounded_quotes(block)
    assert len(quotes) >= 1
    assert all(isinstance(q, str) and q.strip() for q in quotes)


# ---------------------------------------------------------------------------
# Fail-closed behaviour
# ---------------------------------------------------------------------------
def test_ground_fail_closed_when_engine_missing(monkeypatch):
    """If the engine can't be built, grounding returns an unavailable block
    and never raises — the pipeline must proceed ungrounded."""
    monkeypatch.setattr(kb, "_ENGINE", None)
    monkeypatch.setattr(kb, "_build_engine", lambda: None)
    block = kb.ground("anything")
    assert block.mode == "unavailable"
    assert block.empty is True


# ---------------------------------------------------------------------------
# Acquisition context + HITL guardrail
# ---------------------------------------------------------------------------
def test_dkc_preset_is_danielkliewer():
    assert "DanielKliewer.com" in acq.DKC_PRESET.brand
    assert acq.DKC_PRESET.purpose
    assert acq.acquire_cta()


def test_check_copy_flags_forbidden_claim():
    bad = "We guarantee 100% revenue for every customer who buys."
    flags = acq.check_copy(bad)
    assert flags.ok is False
    assert flags.forbidden_claims


def test_check_copy_passes_clean_copy():
    clean = ("Build local-first AI systems that run on your own hardware. "
             "Explore the stack at DanielKliewer.com.")
    flags = acq.check_copy(clean)
    # clean copy has no forbidden claims; sensitive terms may still be flagged
    assert not flags.forbidden_claims


def test_check_copy_fail_open_on_bad_input():
    flags = acq.check_copy("")  # empty -> ok True with a note
    assert flags.ok is True
    assert flags.notes
