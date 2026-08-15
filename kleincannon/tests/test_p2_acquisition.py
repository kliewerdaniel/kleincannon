"""P2 integration tests — acquisition / closed-deal attribution.

These exercise the REAL learning DB + REAL attribution math (no mocks):

  * feature version bumped to v3 (channel + asset_type one-hots present)
  * blended reward falls back to pure engagement when there are no deals
  * reward_conversion normalises + applies confidence down-weighting
  * time-decay multi-touch attribution credits the right experiences
  * last_touch + manual overrides work
  * low-confidence deals contribute nothing (fail-closed guardrail)
  * record_deal persists a real conversion and credits experiences
  * the bandit's seed_prior nudges theta toward offer-relevant features
  * the full loop still trains on blended reward end-to-end
  * funnel_events are append-only and queryable

Run from the kleincannon repo:
    env -u PYTHONPATH -u PYTHONHOME venv/bin/python -m pytest kleincannon/tests/ -q
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kleincannon.learn import learn_config, db, reward, features, engine as eng
from kleincannon.learn import attribution as attr
from kleincannon.learn import agency, trainer as tr
from kleincannon.episode import Episode


HERE = Path(__file__).resolve().parent
SAMPLE_EPISODE = (ROOT / "kleincannon" / "tests" / "sample_episode")
EPISODE = "2026-08-01-why-the-ocean-is-salty"


def _isolate(tmp: Path) -> None:
    learn_config.learn_dir = str(tmp)
    learn_config.db_path = str(tmp / "experiences.db")
    (Path(learn_config.learn_dir) / "learn").mkdir(parents=True, exist_ok=True)
    eng.ENGINE_PATH = tmp / "bandit_model.npz"
    eng.reset_engine()


# ---------------------------------------------------------------------------
# Feature space
# ---------------------------------------------------------------------------
def test_feature_version_is_v3():
    assert features.FEATURE_VERSION == 3


def test_channel_and_asset_type_features_present():
    names = features.feature_names()
    assert "channel_tiktok" in names
    assert "channel_linkedin" in names
    assert "asset_short_video" in names
    assert "asset_article" in names


def test_channel_default_is_tiktok_short_video():
    f = features.extract({})
    assert f["channel_tiktok"] == 1.0
    assert f["asset_short_video"] == 1.0


def test_channel_override_changes_onehot():
    f = features.extract({"channel": "linkedin", "asset_type": "article"})
    assert f["channel_linkedin"] == 1.0 and f["channel_tiktok"] == 0.0
    assert f["asset_article"] == 1.0 and f["asset_short_video"] == 0.0


# ---------------------------------------------------------------------------
# Blended reward
# ---------------------------------------------------------------------------
def test_blend_falls_back_to_engagement_with_no_deals(tmp_path):
    _isolate(tmp_path)
    w_eng, w_conv = reward.blend_weights(attributed_deals=0)
    assert w_eng > 0.0
    assert w_conv == 0.0
    m = {"views": 5000, "likes": 300}
    assert reward.blended_reward(m, deal_value=0) == reward.score(m)


def test_blend_weights_grow_with_deals(tmp_path):
    _isolate(tmp_path)
    w_eng_0, w_conv_0 = reward.blend_weights(attributed_deals=0)
    w_eng_8, w_conv_8 = reward.blend_weights(attributed_deals=8)
    assert w_conv_8 > w_conv_0
    assert w_eng_8 < w_eng_0


def test_reward_conversion_normalises_and_downweights_confidence():
    hi = reward.reward_conversion(1000.0, confidence=1.0)
    lo = reward.reward_conversion(1000.0, confidence=0.5)
    assert hi > 0.0
    assert lo < hi  # lower confidence -> lower credit
    zero = reward.reward_conversion(1000.0, confidence=0.0)
    assert zero == 0.0


# ---------------------------------------------------------------------------
# Attribution math (pure, deterministic)
# ---------------------------------------------------------------------------
def _deal(**kw):
    base = {"deal_id": "d1", "closed_at": time.time(), "value": 1000.0, "offer": "",
            "touchpoints": [], "attributed_experience_ids": [],
            "attribution_method": "time_decay", "confidence": 1.0, "source_ref": "test"}
    base.update(kw)
    return base


def test_time_decay_credits_recent_touchpoints_more():
    now = time.time()
    deal = _deal(touchpoints=[
        {"experience_id": "e_old", "at": now - 28 * 86400},   # ~2 half-lives ago
        {"experience_id": "e_new", "at": now - 1 * 86400},     # ~0 half-lives
    ])
    credits = dict(attr.credit_for(deal))
    assert "e_old" in credits and "e_new" in credits
    assert credits["e_new"] > credits["e_old"]
    # sum of credits == effective value (confidence 1.0)
    assert abs(sum(credits.values()) - 1000.0) < 1e-6


def test_low_confidence_deal_credits_nothing():
    deal = _deal(confidence=0.1, touchpoints=[{"experience_id": "e1", "at": time.time()}])
    assert attr.credit_for(deal) == []


def test_last_touch_attribution():
    deal = _deal(attribution_method="last_touch",
                 attributed_experience_ids=["e_a", "e_b"])
    credits = dict(attr.credit_for(deal))
    assert credits == {"e_b": 1000.0}


def test_manual_attribution_splits_equally():
    deal = _deal(attribution_method="manual",
                 attributed_experience_ids=["e_a", "e_b", "e_c"], confidence=0.5)
    credits = dict(attr.credit_for(deal))
    assert set(credits) == {"e_a", "e_b", "e_c"}
    # each gets value*confidence / 3 == 500/3
    assert abs(credits["e_a"] - 500.0 / 3) < 1e-3


# ---------------------------------------------------------------------------
# Deal persistence + crediting (real DB)
# ---------------------------------------------------------------------------
def test_record_deal_persists_and_credits(tmp_path):
    _isolate(tmp_path)
    store = db.open_db()
    store.add_experience(db.Experience(id="e1", episode_id=EPISODE, platform="tiktok"))
    store.add_experience(db.Experience(id="e2", episode_id=EPISODE, platform="tiktok"))
    store.close()

    res = attr.record_deal(
        deal_id="deal_real", value=2000.0, confidence=1.0,
        touchpoints=[
            {"experience_id": "e1", "at": time.time() - 86400},
            {"experience_id": "e2", "at": time.time()},
        ],
        source_ref="crm:acme")
    assert res["n_credited"] == 2
    assert db.open_db().conversion_count() == 1

    store = db.open_db()
    e2 = store.get_experience("e2")
    assert e2.attributed_value > 0.0  # recent touch got the larger share
    e1 = store.get_experience("e1")
    assert e1.attributed_value > 0.0 and e1.attributed_value < e2.attributed_value
    store.close()


def test_record_deal_skips_unknown_experience(tmp_path):
    _isolate(tmp_path)
    res = attr.record_deal(
        deal_id="d_ghost", value=500.0,
        attributed_experience_ids=["does_not_exist"],
        attribution_method="manual")
    assert res["n_credited"] == 0
    assert db.open_db().conversion_count() == 1  # deal still logged


# ---------------------------------------------------------------------------
# Funnel telemetry (append-only)
# ---------------------------------------------------------------------------
def test_funnel_events_append_only(tmp_path):
    _isolate(tmp_path)
    store = db.open_db()
    store.add_funnel_event("e1", "impression", channel="tiktok")
    store.add_funnel_event("e1", "lead", channel="tiktok", deal_id="d1")
    rows = store.list_funnel_events("e1")
    assert len(rows) == 2
    assert rows[0]["stage"] == "impression"
    store.close()


# ---------------------------------------------------------------------------
# Bandit prior
# ---------------------------------------------------------------------------
def test_seed_prior_nudges_theta():
    b = eng.ContextualBandit()
    before = b.theta.copy()
    # bias educational content upward
    bias = {"content_educational": 1.0, "content_urgency": 0.5}
    b.seed_prior(bias, strength=0.5)
    # theta should have changed
    assert not np.allclose(before, b.theta)
    # the biased feature index should now be > 0 (prior pulled it up)
    names = b.feature_names_
    idx = names.index("content_educational")
    assert b.theta[idx] > 0.0


# ---------------------------------------------------------------------------
# Full loop trains on blended reward
# ---------------------------------------------------------------------------
def test_full_loop_trains_on_blended_reward(tmp_path):
    _isolate(tmp_path)
    # seed ≥ retrain_min_experiences posted experiences with metrics + a deal
    n = max(8, learn_config.retrain_min_experiences)
    store = db.open_db()
    for i in range(n):
        exp = db.Experience(
            id=f"exp{i}", episode_id=EPISODE, platform="tiktok",
            generation_params={"script": f"learn local-first AI {i}",
                               "hook": "you own your models",
                               "channel": "tiktok", "asset_type": "short_video"})
        store.add_experience(exp)
        store.update_experience(f"exp{i}", upload_status="uploaded", posted_at=time.time())
        store.add_snapshot(db.MetricSnapshot(
            experience_id=f"exp{i}", t_offset=0.0, captured_at=time.time(),
            metrics={"views": 3000 * (i + 1), "likes": 200 * (i + 1)}))
    # a real closed deal credited to exp0
    attr.record_deal(deal_id="loop_deal", value=1500.0, confidence=1.0,
                     touchpoints=[{"experience_id": "exp0", "at": time.time()}])
    store.close()

    res = tr.train_from_history()
    assert res.get("trained"), res
    # blended reward must incorporate the attributed value on exp0
    store = db.open_db()
    exp0 = store.get_experience("exp0")
    assert exp0 is not None and exp0.attributed_value > 0.0
    # the trained engine can still predict (non-degenerate)
    mean, _ = eng.get_engine().predict(
        {"script": "local-first sovereignty", "hook": "take back control",
         "channel": "tiktok", "asset_type": "short_video"})
    assert isinstance(mean, float)
    store.close()
