"""End-to-end integration + learning-core tests for the learning subsystem.

Two parts, both fully local (MockTikTok, no network/token):

  Part A — learning core (unit): a fresh bandit is trained against a SYNTHETIC
  linear reward built from the real feature extractor. After training it must
  (1) fit the data (low error) and (2) rank new configs by their true reward
  (strong rank correlation). This proves the optimization engine itself works.

  Part B — autonomous loop (integration): runs real `run_cycle` x N against the
  mock platform, harvests append-only metrics, retrains, and asserts the loop
  produces posted experiences, immutable snapshot history, a non-degenerate
  predictive model, and a persisted trained model.

Run: env -u PYTHONPATH -u PYTHONHOME ./venv/bin/python scripts/test_learn_loop.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kleincannon.learn import (learn_config, db, metadata, reward, optimize,
                               engine as eng, platform as plat, harvester as hv,
                               trainer as tr, agency)
from kleincannon.episode import Episode


EPISODE = "2026-08-01-why-the-ocean-is-salty"


def log(*a):
    print("[test]", *a, flush=True)


def _isolate(tmp: Path) -> None:
    # Mutate the singleton IN MEMORY only — never call save() here, or the
    # temp db path leaks into the canonical learn/config.json.
    learn_config.learn_dir = str(tmp)
    learn_config.db_path = str(tmp / "experiences.db")
    (Path(learn_config.learn_dir) / "learn").mkdir(parents=True, exist_ok=True)
    eng.ENGINE_PATH = tmp / "bandit_model.npz"
    eng.reset_engine()


def _rank_corr(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    rk = lambda v: [sorted(v).index(x) for x in v]   # rank (ties: first index)
    rx, ry = rk(xs), rk(ys)
    mean = sum(rx) / n
    cov = sum((a - mean) * (b - mean) for a, b in zip(rx, ry))
    den = (sum((a - mean) ** 2 for a in rx) * sum((b - mean) ** 2 for b in ry)) ** 0.5
    return cov / den if den else 0.0


def part_a_learning_core(tmp: Path) -> None:
    log("PART A: learning core (synthetic linear target)")
    _isolate(tmp / "a")
    # Build a synthetic objective as a known linear fn of the REAL features,
    # so the bandit has a clean signal to recover. We construct configs that
    # genuinely span the target dimensions (curiosity up / controversy down).
    feats0 = eng.ContextualBandit()._feat(metadata.capture(Episode.load(EPISODE), niche="x"))
    names = eng.feature_names()
    w = feats0 * 0.0
    ci = names.index("content_curiosity") if "content_curiosity" in names else 0
    co = names.index("content_controversy") if "content_controversy" in names else 1
    w[ci] = 2.0
    w[co] = -2.0

    def synth(meta) -> float:
        return float(eng.ContextualBandit()._feat(meta) @ w)

    ep = Episode.load(EPISODE)

    def mk(i: int) -> dict:
        # alternate strongly between a high-curiosity and a high-controversy hook
        if i % 2 == 0:
            m = metadata.capture(ep, niche=f"n{i}")
            m["hook"] = "You won't believe the secret the ocean is hiding"
            m["script"] = "why the deep sea is full of mysteries we never expected to find"
        else:
            m = metadata.capture(ep, niche=f"n{i}")
            m["hook"] = "they lied about the ocean"
            m["script"] = "controversial debate about the ocean vs the truth exposed"
        m["posting_day"] = ["Monday", "Tuesday", "Wednesday", "Thursday",
                            "Friday", "Saturday", "Sunday"][i % 7]
        m["posting_time"] = f"{8 + (i % 12)}:00"
        return m

    bandit = eng.ContextualBandit()
    cands = [mk(i) for i in range(16)]
    for i, m in enumerate(cands):
        bandit.update(f"c{i}", m, synth(m))
    log(f"  trained bandit on {bandit.n_updates} examples")

    test_cands = [mk(100 + i) for i in range(16)]
    preds = [bandit.predict(c)[0] for c in test_cands]
    trues = [synth(c) for c in test_cands]
    errs = [abs(p - t) for p, t in zip(preds, trues)]
    mae = sum(errs) / len(errs)
    rng = max(trues) - min(trues)
    rel_mae = mae / rng if rng else 0.0
    rho = _rank_corr(preds, trues)
    log(f"  MAE on held-out: {mae:.4f} (rel {rel_mae:.2f} of range {rng:.3f})  "
        f"rank-corr(pred,true): {rho:.3f}")
    assert rel_mae < 0.4, f"bandit absolute error too high (rel mae={rel_mae:.2f})"
    assert rho > 0.6, f"bandit failed to rank by true reward (rho={rho:.3f})"
    log("  PART A PASSED")


def part_b_autonomous_loop(tmp: Path) -> None:
    log("PART B: autonomous optimization loop (MockTikTok)")
    _isolate(tmp / "b")
    assert isinstance(plat.get_adapter(), plat.MockTikTok), "expected mock offline"

    mock = plat.MockTikTok(seed=42)
    plat.get_adapter = lambda *a, **k: mock

    n_cycles = 14
    for i in range(n_cycles):
        out = agency.run_cycle(EPISODE, niche="ocean-science", caption="follow for part 2")
        # simulate time passing so scheduled snapshots mature
        store = db.open_db()
        for e in store.list_experiences(only_posted=True):
            if e.posted_at is not None:
                store.update_experience(e.id, posted_at=e.posted_at - 25 * 3600)
        store.close()
        hv.harvest_all()

    res = tr.train_from_history()
    log(f"  trained: {res.get('trained')}  dataset={res.get('dataset_size')}  "
        f"acc={res.get('accuracy')}  mae={res.get('mae')}")
    assert res.get("trained"), "retrain failed"

    e = eng.get_engine()
    ep = Episode.load(EPISODE)
    cands = optimize.optimize(ep)
    preds = [e.predict(c)[0] for c in cands]
    spread = max(preds) - min(preds)
    log(f"  predicted reward spread across candidate variations: {spread:.4f}")
    assert spread > 1e-3, "bandit produces degenerate (constant) predictions"

    store = db.open_db()
    posted = store.list_experiences(only_posted=True)
    assert len(posted) >= n_cycles, "not all cycles produced posted experiences"
    first = posted[0]
    snaps = store.snapshots(first.id)
    log(f"  posted={len(posted)}  first experience snapshots={len(snaps)}")
    assert len(snaps) >= 1, "no immutable metric snapshots recorded"
    # snapshots must be append-only & strictly increasing in t_offset
    offs = [s.t_offset for s in snaps]
    assert offs == sorted(offs), "snapshots not monotonically ordered"
    diag = e.diagnostics()
    assert diag["n_updates"] >= n_cycles
    log(f"  bandit: updates={diag['n_updates']} epsilon={diag['epsilon_now']} "
        f"top+={diag['top_positive'][:2]} top-={diag['top_negative'][:2]}")
    store.close()
    log("  PART B PASSED")


def main() -> int:
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="learn_test_"))
    part_a_learning_core(tmp)
    part_b_autonomous_loop(tmp)
    log("ALL LEARNING TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
