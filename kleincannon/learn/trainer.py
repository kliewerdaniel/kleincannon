"""Continuous retraining + model management.

The bandit learns *online* (one `update` per finished experience), but we also
periodically retrain from scratch on the full experience DB so the model can't
drift or get stuck on stale weights. Retraining is gated by `retrain_mode`
(every_n_experiences | nightly) and `retrain_min_experiences`.

Each retrain: archives the current `.npz` model (rollback window
`retrain_keep_last`), fits a fresh ContextualBandit over all posted
experiences, evaluates a leave-one-out prediction error for the dashboard, and
writes a row to the `experiments` table with full provenance.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from . import learn_config
from . import db
from . import engine as eng
from . import reward
from . import features

def engine_path() -> Path:
    return Path(learn_config.learn_dir) / "bandit_model.npz"


def archive_dir() -> Path:
    return Path(learn_config.learn_dir) / "models"


def _archive_current() -> None:
    ep = engine_path()
    if not ep.exists():
        return
    ad = archive_dir()
    ad.mkdir(parents=True, exist_ok=True)
    import shutil
    stamp = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy2(ep, ad / f"bandit_model_{stamp}.npz")
    # trim to keep window
    files = sorted(ad.glob("bandit_model_*.npz"), reverse=True)
    for old in files[learn_config.retrain_keep_last:]:
        old.unlink()


def train_from_history() -> dict[str, Any]:
    """Fit a fresh bandit over all posted experiences + their latest metrics.

    Returns training stats. On success the new engine is persisted to
    ENGINE_PATH and an experiment row is written.
    """
    store = db.open_db()
    posted = store.list_experiences(only_posted=True)
    store.close()

    if len(posted) < learn_config.retrain_min_experiences:
        return {"trained": False,
                "reason": f"need >= {learn_config.retrain_min_experiences} posted experiences",
                "have": len(posted)}

    fresh = eng.ContextualBandit()
    for e in posted:
        meta = e.generation_params
        # last metric snapshot is the outcome
        m = db.open_db().latest_metric(e.id) or {}
        if not m:
            continue
        r = reward.score(m)
        fresh.update(e.id, meta, r)

    _archive_current()
    fresh_path = engine_path()
    fresh.save(str(fresh_path))

    # leave-one-out cross-val prediction error (for the dashboard)
    errs = []
    for e in posted:
        m = db.open_db().latest_metric(e.id) or {}
        if not m:
            continue
        goal = reward.score(m)
        pred, _ = fresh.predict(e.generation_params)
        errs.append(abs(pred - goal))
    mae = sum(errs) / len(errs) if errs else 0.0

    # record experiment
    store = db.open_db()
    version = store.save_model("contextual_bandit",
                               fresh.feature_version, {"n_updates": fresh.n_updates},
                               active=True)
    store.save_experiment(
        model_version=version, dataset_size=len(posted),
        hyperparameters={
            "lambda": fresh.lam, "alpha": fresh.alpha,
            "epsilon_initial": fresh.eps0, "epsilon_decay": fresh.eps_decay,
        },
        feature_set=f"features_v{fresh.feature_version}",
        reward_def=learn_config.active_reward_preset,
        accuracy=round(1.0 - min(1.0, mae), 4),
        validation_metrics={"mae": round(mae, 5),
                            "n": len(errs),
                            "rmse": round((sum(e * e for e in errs) / len(errs)) ** 0.5, 5) if errs else 0.0},
        prediction_error=round(mae, 5),
    )
    store.close()

    # refresh the process-wide engine singleton so live calls use new weights
    eng.reset_engine().load(str(engine_path()))

    return {"trained": True, "model_version": version,
            "dataset_size": len(posted), "mae": round(mae, 5),
            "accuracy": round(1.0 - min(1.0, mae), 4),
            "feature_version": fresh.feature_version,
            "epsilon_now": fresh.diagnostics()["epsilon_now"]}


def maybe_retrain() -> dict[str, Any]:
    """Called after each cycle; retrains only when the gate is satisfied."""
    store = db.open_db()
    posted = store.list_experiences(only_posted=True)
    count = len(posted)
    store.close()
    if count < learn_config.retrain_min_experiences:
        return {"trained": False, "reason": "below min experiences"}
    if learn_config.retrain_mode == "every_n_experiences":
        if count % learn_config.retrain_every_n != 0:
            return {"trained": False, "reason": "not on retrain boundary"}
    # nightly mode would be triggered by a scheduler; here we just train when due
    return train_from_history()


def rollback_to(version: int) -> dict[str, Any]:
    """Restore a previously archived model version as active."""
    store = db.open_db()
    m = store.model(version)
    if not m:
        store.close()
        return {"rolled_back": False, "reason": f"no model version {version}"}
    store.set_active_model(version)
    store.close()
    # copy archive back if present
    arch = archive_dir() / f"bandit_model_{version}.npz"
    ep = engine_path()
    if arch.exists() and ep.exists():
        import shutil
        shutil.copy2(arch, ep)
    eng.reset_engine().load(str(ep))
    return {"rolled_back": True, "version": version}


def history() -> dict[str, Any]:
    store = db.open_db()
    models = store.list_models()
    experiments = store.list_experiments()
    store.close()
    return {"models": models, "experiments": experiments}
