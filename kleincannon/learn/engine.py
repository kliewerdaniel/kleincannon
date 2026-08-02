"""Learning engine — the autonomous optimization core.

Implements a **Contextual Bandit** (LinUCB-style): one ridge-regression
estimator approximates reward as a linear function of the content/video/
posting/creator feature vector, and a per-arm uncertainty term drives
exploration. This is a strong, well-understood baseline for "which generation
parameters should we try next" that is cheap, online, and interpretable — and
it runs entirely offline (no GPU).

Everything sits behind interfaces:
  * `LearningEngine` protocol — `update(exp_id, meta, reward)`,
    `predict(meta) -> (mean, std)`, `best(metas) -> meta`, `save()/load()`.
  * The concrete `ContextualBandit` is selected by learn_config.learner_type,
    so a PPO/DQN/learned-embedding engine can be dropped in later without
    touching callers (candidate optimizer, retrainer, dashboard).

No magic numbers: lambda/alpha/epsilon decay all come from learn_config.
"""
from __future__ import annotations

import json
import time
from typing import Any, Protocol, runtime_checkable

import numpy as np

from . import learn_config
from .features import extract, feature_names, feature_dim


@runtime_checkable
class LearningEngine(Protocol):
    def update(self, exp_id: str, meta: dict[str, Any], reward: float) -> None: ...
    def predict(self, meta: dict[str, Any]) -> tuple[float, float]: ...
    def best(self, metas: list[dict[str, Any]]) -> tuple[dict[str, Any], float, float]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> bool: ...


class ContextualBandit:
    """LinUCB-style contextual bandit with ridge regression + uncertainty.

    A single linear reward model is shared across all "arms" (each candidate
    generation config is a context, not a separate arm — this is the
    contextual setting where we generalise across configs via the feature
    vector). Exploration = upper-confidence-bound on the prediction.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or learn_config
        self.dim = feature_dim()
        self.lam = float(self.cfg.bandit_lambda)
        self.alpha = float(self.cfg.bandit_alpha)
        self.eps0 = float(self.cfg.bandit_epsilon_initial)
        self.eps_min = float(self.cfg.bandit_epsilon_min)
        self.eps_decay = float(self.cfg.bandit_epsilon_decay)
        self.feature_version = int(self.cfg.bandit_feature_version)
        # ridge A (dim x dim) + b (dim).  A_inv maintained for O(dim^2) updates.
        self.A = self.lam * np.eye(self.dim, dtype=float)
        self.A_inv = np.linalg.inv(self.A)
        self.b = np.zeros(self.dim, dtype=float)
        self.theta = np.zeros(self.dim, dtype=float)
        self.n_updates = 0
        self.feature_names_ = feature_names()

    # ---- core math ----
    def _feat(self, meta: dict[str, Any]) -> np.ndarray:
        f = extract(meta)
        return np.array([f.get(k, 0.0) for k in self.feature_names_], dtype=float)

    def predict(self, meta: dict[str, Any]) -> tuple[float, float]:
        x = self._feat(meta)
        mean = float(self.theta @ x)
        # predictive variance: x^T A^{-1} x  (LinUCB uncertainty)
        var = float(x @ self.A_inv @ x)
        std = float(np.sqrt(max(0.0, var)))
        return mean, std

    def _epsilon(self) -> float:
        # decay with number of updates, floored
        return max(self.eps_min, self.eps0 * (self.eps_decay ** self.n_updates))

    def best(self, metas: list[dict[str, Any]]) -> tuple[dict[str, Any], float, float]:
        """Upper-confidence-bound selection across candidate metas.

        Returns (chosen_meta, mean, std). With probability epsilon we pick
        uniformly at random (pure exploration); otherwise we pick the highest
        UCB score.
        """
        if not metas:
            raise ValueError("no candidate metas supplied to best()")
        if np.random.rand() < self._epsilon():
            idx = int(np.random.randint(0, len(metas)))
            m, s = self.predict(metas[idx])
            return metas[idx], m, s
        scores = []
        for meta in metas:
            mean, std = self.predict(meta)
            scores.append((mean + self.alpha * std, mean, std, meta))
        scores.sort(key=lambda t: t[0], reverse=True)
        _, mean, std, meta = scores[0]
        return meta, mean, std

    def update(self, exp_id: str, meta: dict[str, Any], reward: float) -> None:
        x = self._feat(meta)
        # sherman-morrison update of A_inv
        Ax = self.A @ x
        denom = 1.0 + float(x @ self.A_inv @ x)
        self.A_inv = self.A_inv - np.outer(self.A_inv @ x, x @ self.A_inv) / denom
        self.A = self.A + np.outer(x, x)
        self.b = self.b + reward * x
        self.theta = self.A_inv @ self.b
        self.n_updates += 1

    # ---- persistence ----
    def save(self, path: str) -> None:
        np.savez(path, A=self.A, A_inv=self.A_inv, b=self.b, theta=self.theta,
                 n_updates=np.array(self.n_updates),
                 feature_version=np.array(self.feature_version))
        # companion json for human-inspectable metadata
        Path(path).with_suffix(".json").write_text(json.dumps({
            "dim": self.dim, "n_updates": self.n_updates,
            "feature_version": self.feature_version,
            "epsilon_now": self._epsilon(),
            "saved_at": time.time(),
        }, indent=2))

    def load(self, path: str) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        data = np.load(path)
        # feature-version mismatch => refuse (model trained on old features)
        fv = int(data["feature_version"])
        if fv != self.feature_version:
            return False
        self.A = data["A"]
        self.A_inv = data["A_inv"]
        self.b = data["b"]
        self.theta = data["theta"]
        self.n_updates = int(data["n_updates"])
        return True

    def weights(self) -> dict[str, float]:
        """Return feature_name -> learned weight for dashboard interpretability."""
        return {k: round(float(v), 5) for k, v in zip(self.feature_names_, self.theta)}

    def diagnostics(self) -> dict[str, Any]:
        return {
            "type": "contextual_bandit",
            "n_updates": self.n_updates,
            "epsilon_now": round(self._epsilon(), 4),
            "feature_dim": self.dim,
            "feature_version": self.feature_version,
            "alpha": self.alpha,
            "lambda": self.lam,
            "confidence": min(1.0, self.n_updates / max(1, self.cfg.retrain_min_experiences)),
            "top_positive": sorted(self.weights().items(), key=lambda kv: kv[1], reverse=True)[:5],
            "top_negative": sorted(self.weights().items(), key=lambda kv: kv[1])[:5],
        }


def build_engine(cfg=None) -> LearningEngine:
    cfg = cfg or learn_config
    if cfg.learner_type == "contextual_bandit":
        return ContextualBandit(cfg)
    # Future: elif cfg.learner_type == "ppo": return PPOEngine(cfg)
    raise ValueError(f"unknown learner_type: {cfg.learner_type}")


# lazily resolve Path for the engine checkpoint in the learn dir.
from pathlib import Path  # noqa: E402
ENGINE_PATH = Path(learn_config.learn_dir) / "bandit_model.npz"


# A process-wide engine instance, loaded lazily from disk.
_engine: ContextualBandit | None = None


def get_engine() -> ContextualBandit:
    global _engine
    if _engine is None:
        _engine = ContextualBandit()
        _engine.load(str(ENGINE_PATH))
    return _engine


def reset_engine() -> ContextualBandit:
    global _engine
    _engine = ContextualBandit()
    return _engine
