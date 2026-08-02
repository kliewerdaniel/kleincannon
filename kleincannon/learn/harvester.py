"""Metrics harvester — measures outcomes over time.

For each posted experience it pulls metrics from the active platform adapter
at the append-only `poll_schedule` offsets (1h, 6h, 1d, 3d, 7d, 14d, 30d by
default). Each pull is stored as an *immutable* snapshot — never overwritten —
so an experience accumulates a full performance history. The latest snapshot
is what the reward engine scores.

`harvest_once` is the synchronous single-shot used right after upload and by
the CLI. `Harvester` is the background worker that wakes on
`harvest_loop_interval` and checks whether any experience is due for a new
snapshot.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from . import learn_config
from . import db
from . import platform as plat
from . import reward


def _due_offsets(posted_at: float, taken: set[float], now: float) -> list[float]:
    due = []
    for off in learn_config.poll_schedule:
        if off in taken:
            continue
        if now >= posted_at + off:
            due.append(off)
    return due


def harvest_once(experience_id: str) -> dict[str, Any]:
    """Pull the currently-due snapshot(s) for one experience and store them.

    Returns a summary of what was captured (and whether the experience is now
    fully harvested per the schedule).
    """
    store = db.open_db()
    exp = store.get_experience(experience_id)
    if not exp or exp.posted_at is None:
        store.close()
        return {"experience_id": experience_id, "captured": 0,
                "reason": "not posted yet"}
    adapter = plat.get_adapter()
    taken = {snap.t_offset for snap in store.snapshots(experience_id)}
    now = time.time()
    due = _due_offsets(exp.posted_at, taken, now)
    captured = 0
    last: dict[str, float] = {}
    for off in due:
        if len(store.snapshots(experience_id)) >= learn_config.harvest_max_snapshots:
            break
        m = adapter.get_metrics(exp.video_id) if exp.video_id else {}
        if m:
            ok = store.add_snapshot(db.MetricSnapshot(
                experience_id=experience_id, t_offset=off, captured_at=now, metrics=m))
            if ok:
                captured += 1
                last = m
    # update the experience's stored reward from the latest snapshot
    if last:
        r = reward.score(last)
        store.update_experience(experience_id, reward=round(r, 5))
    store.close()
    return {"experience_id": experience_id, "captured": captured,
            "latest_reward": round(reward.score(last), 5) if last else 0.0,
            "fully_harvested": not _due_offsets(exp.posted_at,
                                               taken | set(due), time.time())}


def harvest_all() -> dict[str, Any]:
    """Harvest every posted experience that is due. Used by the background
    loop and the `kc learn harvest` command."""
    store = db.open_db()
    posted = store.list_experiences(only_posted=True)
    store.close()
    results = []
    total = 0
    for e in posted:
        res = harvest_once(e.id)
        total += res.get("captured", 0)
        results.append(res)
    return {"harvested_experiences": len(results), "snapshots_captured": total,
            "results": results}


class Harvester:
    """Background worker. `start()` spins a daemon thread that wakes every
    `harvest_loop_interval` seconds and harvests due experiences."""

    def __init__(self):
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._stop.set()

    def _loop(self) -> None:
        while self._running:
            try:
                harvest_all()
            except Exception:
                # keep the loop alive; bad network on one cycle shouldn't kill it
                pass
            self._stop.wait(learn_config.harvest_loop_interval)

    def is_running(self) -> bool:
        return self._running


# process-wide singleton
_harvester = Harvester()


def get_harvester() -> Harvester:
    return _harvester
