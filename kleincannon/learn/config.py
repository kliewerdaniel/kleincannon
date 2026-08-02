"""Learning subsystem configuration.

Single source of truth for every tunable knob in the optimization engine.
No magic numbers live in subsystem code — they all resolve to learn.json
(or to config.py for the few shared pipeline defaults). The file is created
with sane defaults on first import and re-loaded whenever the user edits it,
so `kc learn init` (or just running any learn command) keeps it in sync.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .. import config

# Where the learning subsystem stores everything. Kept under the project so
# it stays machine-local and git-ignored (media + the DB are large/machine-specific).
LEARN_DIR = config.ROOT / "learn"
DEFAULT_CONFIG_PATH = LEARN_DIR / "config.json"

# Append-only poll schedule (seconds). Matches the spec's suggested cadence.
DEFAULT_POLL_SCHEDULE = [3600, 6 * 3600, 24 * 3600, 3 * 86400, 7 * 86400,
                         14 * 86400, 30 * 86400]

# Reward presets. Keys are metric names; weights are relative and normalised
# by the reward engine. "balanced_growth" is the default active preset.
DEFAULT_REWARD_PRESETS = {
    "max_views": {"views": 1.0},
    "max_followers": {"followers_gained": 1.0},
    "highest_engagement": {"likes": 0.4, "comments": 0.25, "shares": 0.25,
                           "saves": 0.10},
    "highest_watch_time": {"watch_time": 0.6, "completion_rate": 0.4},
    "balanced_growth": {"views": 0.30, "likes": 0.18, "comments": 0.14,
                        "shares": 0.14, "saves": 0.12, "followers_gained": 0.12},
}


@dataclass
class LearnConfig:
    # ---- storage ----
    db_path: str = str(LEARN_DIR / "experiences.db")
    learn_dir: str = str(LEARN_DIR)
    media_root: str = str(LEARN_DIR / "media")          # archived candidates

    # ---- platform ----
    # Active platform adapter name. Only "tiktok" ships; the adapter layer is
    # built so "youtube_shorts" / "instagram_reels" / etc. can be added later
    # without touching anything else.
    platform: str = "tiktok"
    # Credentials live in the OS keychain / .env, never in this file.
    # These point the adapters at where the secrets are.
    client_key_env: str = "TIKTOK_CLIENT_KEY"
    client_secret_env: str = "TIKTOK_CLIENT_SECRET"
    token_path: str = str(LEARN_DIR / "tiktok_token.json")

    # ---- upload ----
    upload_auto_publish: bool = False       # require explicit decision normally
    upload_retries: int = 3
    upload_retry_base_delay: float = 5.0    # seconds; exponential backoff
    upload_rate_limit_per_hour: int = 10
    default_privacy: str = "SELF_ONLY"      # SELF_ONLY | PUBLIC_TO_ALL
    default_title: str = ""
    default_description: str = ""
    # Hashtags are posted WITH the video (needed for reach); CTA is NOT burned
    # into the video per the no-auto-burn rule, but may be in the post caption.
    default_hashtags: list[str] = field(default_factory=list)
    default_caption: str = ""

    # ---- metrics harvester ----
    # Append-only poll offsets (seconds from upload time). A snapshot is taken
    # at each offset if it hasn't been already. Never overwritten.
    poll_schedule: list[float] = field(default_factory=lambda: list(DEFAULT_POLL_SCHEDULE))
    harvest_loop_interval: float = 1800.0   # background worker wake-up (30 min)
    harvest_max_snapshots: int = 64         # safety cap per experience

    # ---- reward ----
    active_reward_preset: str = "balanced_growth"
    reward_presets: dict[str, dict[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_REWARD_PRESETS.items()})
    # Reward is normalised to this per-metric reference so different metrics
    # (views vs followers) are comparable. These are *priors*; the harvester
    # also tracks empirical per-metric 95th percentiles and can auto-scale.
    reward_reference: dict[str, float] = field(default_factory=lambda: {
        "views": 5000.0, "likes": 300.0, "comments": 50.0, "shares": 30.0,
        "saves": 40.0, "followers_gained": 20.0, "watch_time": 200.0,
        "completion_rate": 0.5, "profile_visits": 100.0,
    })

    # ---- learning engine (Contextual Bandit) ----
    learner_type: str = "contextual_bandit"  # interface allows future ppo/dqn/...
    bandit_lambda: float = 1.0             # ridge regularisation
    bandit_alpha: float = 1.0             # exploration scaling (uncertainty weight)
    bandit_epsilon_initial: float = 0.40  # 40% random exploration when we know little
    bandit_epsilon_min: float = 0.05       # floor as confidence grows
    bandit_epsilon_decay: float = 0.97    # per-experience decay (slower = more exploring)
    bandit_min_confidence: float = 0.5     # below this, lean exploration
    bandit_feature_version: int = 1        # bump to invalidate old model weights

    # ---- candidate optimization ----
    candidate_count: int = 20             # generate N, publish the top-1
    candidate_publish_top_k: int = 1       # publish only the best
    candidate_archive_rest: bool = True    # keep losers for training

    # ---- evolutionary prompt mutation ----
    mutation_variations: list[str] = field(default_factory=lambda: [
        "shorter", "longer", "more_emotional", "more_curiosity",
        "stronger_hook", "more_urgency", "different_cta", "different_pacing",
    ])
    mutation_temperature: float = 0.9
    mutation_keep_top_k_parents: int = 5   # best experiences seed mutations
    mutation_lineage_depth: int = 8

    # ---- continuous retraining ----
    retrain_mode: str = "every_n_experiences"  # or "nightly"
    retrain_every_n: int = 100
    retrain_archive_models: bool = True
    retrain_keep_last: int = 5              # rollback window
    retrain_min_experiences: int = 8        # need data before first model

    # ---- logging ----
    log_level: str = "INFO"

    # ---- runtime (not persisted) ----
    _path: Path | None = field(default=None, repr=False)

    def save(self) -> Path:
        path = self._path or DEFAULT_CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data.pop("_path", None)
        # JSON can't do the Path objects directly — they were stored as str.
        path.write_text(json.dumps(data, indent=2))
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "LearnConfig":
        path = path or DEFAULT_CONFIG_PATH
        if not path.exists():
            cfg = cls(_path=path)
            cfg.save()
            return cfg
        raw = json.loads(path.read_text())
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in raw.items() if k in known and k != "_path"}
        cfg = cls(**filtered)
        cfg._path = path
        return cfg

    # Convenience: resolve a metric weight dict for the active preset.
    def reward_weights(self) -> dict[str, float]:
        return self.reward_presets.get(
            self.active_reward_preset,
            self.reward_presets["balanced_growth"],
        )

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def get_learn_path() -> Path:
    return DEFAULT_CONFIG_PATH


# Module-level singleton, lazily created/loaded. Stages import `learn_config`.
def _make() -> LearnConfig:
    LEARN_DIR.mkdir(parents=True, exist_ok=True)
    return LearnConfig.load(DEFAULT_CONFIG_PATH)


learn_config = _make()
