"""Experience Database — the learning subsystem's durable memory.

One uploaded video == one experience row. Metrics are appended as immutable
snapshots (never updated in place), so an experience builds a complete
performance *history* over time. Models and experiments are also stored with
full provenance, and old models are archived so we can roll back.

Dependency-free: sqlite3 (stdlib) only.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from . import learn_config


@dataclass
class MetricSnapshot:
    experience_id: str
    t_offset: float            # seconds since posted_at
    captured_at: float         # unix ts
    metrics: dict[str, float]  # raw platform metrics


@dataclass
class Experience:
    id: str
    episode_id: str
    platform: str
    poster_account: str = ""
    niche: str = ""
    video_id: Optional[str] = None
    upload_status: str = "pending"     # pending | uploaded | failed
    posted_at: Optional[float] = None
    generation_params: dict = field(default_factory=dict)
    upload_info: dict = field(default_factory=dict)
    reward: float = 0.0
    reward_preset: str = ""
    chosen: bool = False               # published candidate vs archived
    parent_id: Optional[str] = None    # lineage
    variation: str = ""                # mutation type applied
    archived_path: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row: dict) -> "Experience":
        return cls(
            id=row["id"], episode_id=row["episode_id"], platform=row["platform"],
            poster_account=row.get("poster_account", ""), niche=row.get("niche", ""),
            video_id=row.get("video_id"), upload_status=row.get("upload_status", "pending"),
            posted_at=row.get("posted_at"),
            generation_params=json.loads(row.get("generation_params") or "{}"),
            upload_info=json.loads(row.get("upload_info") or "{}"),
            reward=row.get("reward", 0.0), reward_preset=row.get("reward_preset", ""),
            chosen=bool(row.get("chosen", 0)), parent_id=row.get("parent_id"),
            variation=row.get("variation", ""), archived_path=row.get("archived_path"),
            created_at=row.get("created_at", time.time()),
        )


class ExperienceDB:
    """Thin, append-friendly SQLite wrapper. No ORM, no surprises."""

    def __init__(self, db_path: str | Path | None = None):
        self.path = Path(db_path or learn_config.db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._migrate()

    def _migrate(self) -> None:
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS experiences (
            id TEXT PRIMARY KEY,
            episode_id TEXT,
            platform TEXT,
            poster_account TEXT,
            niche TEXT,
            video_id TEXT,
            upload_status TEXT,
            posted_at REAL,
            generation_params TEXT,
            upload_info TEXT,
            reward REAL,
            reward_preset TEXT,
            chosen INTEGER,
            parent_id TEXT,
            variation TEXT,
            archived_path TEXT,
            created_at REAL
        );
        CREATE TABLE IF NOT EXISTS metric_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experience_id TEXT,
            t_offset REAL,
            captured_at REAL,
            metrics TEXT,
            UNIQUE(experience_id, t_offset)
        );
        CREATE TABLE IF NOT EXISTS models (
            version INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT,
            feature_version INTEGER,
            trained_at REAL,
            params TEXT,
            active INTEGER
        );
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version INTEGER,
            training_date TEXT,
            dataset_size INTEGER,
            hyperparameters TEXT,
            feature_set TEXT,
            reward_def TEXT,
            accuracy REAL,
            validation_metrics TEXT,
            prediction_error REAL
        );
        CREATE INDEX IF NOT EXISTS idx_snap_exp ON metric_snapshots(experience_id);
        CREATE INDEX IF NOT EXISTS idx_exp_posted ON experiences(posted_at);
        """)
        self.conn.commit()

    # ---- experiences ----
    def add_experience(self, exp: Experience) -> Experience:
        if not exp.id:
            exp.id = uuid.uuid4().hex[:16]
        self.conn.execute(
            """INSERT OR REPLACE INTO experiences
               (id, episode_id, platform, poster_account, niche, video_id,
                upload_status, posted_at, generation_params, upload_info, reward,
                reward_preset, chosen, parent_id, variation, archived_path, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (exp.id, exp.episode_id, exp.platform, exp.poster_account, exp.niche,
             exp.video_id, exp.upload_status, exp.posted_at,
             json.dumps(exp.generation_params), json.dumps(exp.upload_info),
             exp.reward, exp.reward_preset, int(exp.chosen), exp.parent_id,
             exp.variation, exp.archived_path, exp.created_at),
        )
        self.conn.commit()
        return exp

    def update_experience(self, exp_id: str, **fields) -> None:
        if not fields:
            return
        import json as _json
        vals: list[Any] = []
        sets_parts: list[str] = []
        for k, v in fields.items():
            if isinstance(v, (dict, list)):
                v = _json.dumps(v)
            sets_parts.append(f"{k}=?")
            vals.append(v)
        sets = ", ".join(sets_parts)
        vals.append(exp_id)
        self.conn.execute(f"UPDATE experiences SET {sets} WHERE id=?", vals)
        self.conn.commit()

    def get_experience(self, exp_id: str) -> Optional[Experience]:
        row = self.conn.execute("SELECT * FROM experiences WHERE id=?",
                                (exp_id,)).fetchone()
        return Experience.from_row(dict(row)) if row else None

    def get_by_episode(self, episode_id: str) -> list[Experience]:
        rows = self.conn.execute(
            "SELECT * FROM experiences WHERE episode_id=? ORDER BY created_at",
            (episode_id,)).fetchall()
        return [Experience.from_row(dict(r)) for r in rows]

    def list_experiences(self, only_posted: bool = False,
                         only_chosen: bool = False,
                         limit: int = 500) -> list[Experience]:
        q = "SELECT * FROM experiences WHERE 1=1"
        args: list[Any] = []
        if only_posted:
            q += " AND posted_at IS NOT NULL"
        if only_chosen:
            q += " AND chosen=1"
        q += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        rows = self.conn.execute(q, args).fetchall()
        return [Experience.from_row(dict(r)) for r in rows]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]

    # ---- metric snapshots (APPEND ONLY) ----
    def add_snapshot(self, snap: MetricSnapshot) -> bool:
        """Append a snapshot. Returns False if one at this t_offset exists
        (snapshots are immutable history — never overwritten)."""
        try:
            self.conn.execute(
                """INSERT INTO metric_snapshots (experience_id, t_offset, captured_at, metrics)
                   VALUES (?,?,?,?)""",
                (snap.experience_id, snap.t_offset, snap.captured_at,
                 json.dumps(snap.metrics)),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def snapshots(self, exp_id: str) -> list[MetricSnapshot]:
        rows = self.conn.execute(
            "SELECT * FROM metric_snapshots WHERE experience_id=? ORDER BY t_offset",
            (exp_id,)).fetchall()
        return [MetricSnapshot(r["experience_id"], r["t_offset"], r["captured_at"],
                               json.loads(r["metrics"])) for r in rows]

    def latest_metric(self, exp_id: str) -> Optional[dict[str, float]]:
        rows = self.conn.execute(
            "SELECT metrics FROM metric_snapshots WHERE experience_id=? "
            "ORDER BY t_offset DESC LIMIT 1", (exp_id,)).fetchone()
        return json.loads(rows["metrics"]) if rows else None

    # ---- models + experiments ----
    def save_model(self, mtype: str, feature_version: int, params: dict,
                   active: bool = True) -> int:
        if active:
            self.conn.execute("UPDATE models SET active=0")
        cur = self.conn.execute(
            "INSERT INTO models (type, feature_version, trained_at, params, active) "
            "VALUES (?,?,?,?,?)",
            (mtype, feature_version, time.time(), json.dumps(params), int(active)))
        self.conn.commit()
        return int(cur.lastrowid) if cur.lastrowid is not None else -1

    def active_model(self) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM models WHERE active=1 "
                                "ORDER BY version DESC LIMIT 1").fetchone()
        if not row:
            return None
        d = dict(row)
        d["params"] = json.loads(d["params"])
        return d

    def model(self, version: int) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM models WHERE version=?",
                                (version,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["params"] = json.loads(d["params"])
        return d

    def set_active_model(self, version: int) -> None:
        self.conn.execute("UPDATE models SET active=0")
        self.conn.execute("UPDATE models SET active=1 WHERE version=?", (version,))
        self.conn.commit()

    def list_models(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM models ORDER BY version DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["params"] = json.loads(d["params"])
            out.append(d)
        return out

    def save_experiment(self, model_version: int, dataset_size: int,
                        hyperparameters: dict, feature_set: str, reward_def: str,
                        accuracy: float, validation_metrics: dict,
                        prediction_error: float) -> int:
        import datetime as _dt
        cur = self.conn.execute(
            """INSERT INTO experiments
               (model_version, training_date, dataset_size, hyperparameters,
                feature_set, reward_def, accuracy, validation_metrics, prediction_error)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (model_version, _dt.datetime.now().isoformat(timespec="seconds"),
             dataset_size, json.dumps(hyperparameters), feature_set, reward_def,
             accuracy, json.dumps(validation_metrics), prediction_error))
        self.conn.commit()
        return int(cur.lastrowid) if cur.lastrowid is not None else -1

    def list_experiments(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM experiments ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["hyperparameters"] = json.loads(d["hyperparameters"])
            d["validation_metrics"] = json.loads(d["validation_metrics"])
            out.append(d)
        return out

    def close(self) -> None:
        self.conn.close()


def open_db() -> ExperienceDB:
    return ExperienceDB(learn_config.db_path)
