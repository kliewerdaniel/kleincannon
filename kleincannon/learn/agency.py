"""Autonomous research agent — the loop controller.

This is where the spec's "autonomous research agent" lives: an agent that
  * continuously experiments (generates candidate configs, uploads one),
  * measures outcomes (harvests metrics over the append-only schedule),
  * discovers successful strategies (the bandit learns feature->reward),
  * learns from historical data (retrains on the full experience DB),
  * predicts future performance (engine.predict),
  * optimizes generation parameters (optimize.optimize),
  * improves without manual intervention (the harvester + retrainer are
    scheduled background tasks).

It is deliberately a thin orchestrator over the already-building-block modules:
metadata, features, reward, engine, optimize, platform, db, harvester, trainer.
It never mutates the generation pipeline itself.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..episode import Episode
from . import learn_config
from . import db
from . import metadata
from . import reward
from . import optimize
from . import engine as eng
from . import platform as plat
from . import harvester as hv
from . import trainer as tr


# ---------------------------------------------------------------------------
# Experience lifecycle
# ---------------------------------------------------------------------------
def create_experience(episode_id: str, *,
                      niche: str = "", hashtags=None, caption: str = "",
                      parent_id: str | None = None, **extra) -> db.Experience:
    """Capture metadata + a fresh DB row for an episode being prepared for upload."""
    ep = Episode.load(episode_id)
    meta = metadata.capture(ep, niche=niche, hashtags=hashtags,
                            caption=caption, **extra)
    store = db.open_db()
    exp = db.Experience(
        id="", episode_id=episode_id, platform=learn_config.platform,
        poster_account=learn_config.get("_poster_account", "") or "default",
        niche=niche or ep.purpose, generation_params=meta,
        reward_preset=learn_config.active_reward_preset,
        parent_id=parent_id, variation="",
    )
    exp = store.add_experience(exp)
    store.close()
    return exp


def prepare_candidates(episode_id: str, *, parent_id: str | None = None,
                       niche: str = "", hashtags=None, caption: str = "",
                       **extra) -> list[dict[str, Any]]:
    """Generate the candidate population for an episode and persist them as
    archived (not-yet-chosen) experiences so losers still train the model."""
    ep = Episode.load(episode_id)
    candidates = optimize.optimize(ep, parent_id=parent_id, niche=niche,
                                   hashtags=hashtags, caption=caption, **extra)
    store = db.open_db()
    for i, c in enumerate(candidates):
        exp = db.Experience(
            id="", episode_id=episode_id, platform=learn_config.platform,
            poster_account=learn_config.get("_poster_account", "") or "default",
            niche=niche or ep.purpose, generation_params=c,
            reward_preset=learn_config.active_reward_preset,
            parent_id=parent_id, variation=(c.get("variation") or ""),
            chosen=False,
        )
        store.add_experience(exp)
    store.close()
    return candidates


def publish_best(episode_id: str, candidates: list[dict[str, Any]],
                 *, mp4: str | Path | None = None, privacy: str | None = None,
                 niche: str = "") -> dict[str, Any]:
    """Upload the best candidate to the active platform, mark it chosen, and
    seed the metrics harvester.

    The *publish decision* goes through the learning engine's `best()` so that
    exploration (epsilon / upper-confidence-bound) actually happens: an
    untrained bandit will try varied arms instead of always picking the first
    sorted candidate. This is what makes the experience set diverse enough to
    learn from. Returns the upload result + chosen meta.
    """
    engine = eng.get_engine()
    chosen, mean, std = engine.best(candidates)
    # ensure the chosen candidate has its id + posting stamps for later matching
    chosen = dict(chosen)
    adapter = plat.get_adapter()
    mp4 = Path(mp4) if mp4 else (Episode.load(episode_id).dir / f"{episode_id}.mp4")
    if not mp4.exists():
        raise FileNotFoundError(f"no mp4 at {mp4}")

    # stamp posting time/day
    chosen = metadata.embed_posting_time(dict(chosen), time.time())
    chosen["privacy"] = privacy or learn_config.default_privacy
    chosen["title"] = chosen.get("title") or Episode.load(episode_id).topic

    result = adapter.upload_video(mp4, chosen)

    store = db.open_db()
    # find the experience row that matches this candidate id
    exp = None
    for cand_exp in store.get_by_episode(episode_id):
        if cand_exp.generation_params.get("_candidate_id") == chosen.get("_candidate_id"):
            exp = cand_exp
            break
    if exp is None:
        exp = db.Experience(id="", episode_id=episode_id,
                            platform=learn_config.platform,
                            niche=niche or Episode.load(episode_id).purpose,
                            generation_params=chosen)
        exp = store.add_experience(exp)
    store.update_experience(exp.id,
                            upload_status="uploaded",
                            video_id=result.get("video_id"),
                            posted_at=time.time(),
                            chosen=True,
                            generation_params=chosen)
    store.close()
    return {"upload": result, "chosen": chosen, "experience_id": exp.id}


# ---------------------------------------------------------------------------
# One full autonomous cycle: prepare -> publish -> schedule harvest
# ---------------------------------------------------------------------------
def run_cycle(episode_id: str, *, niche: str = "", hashtags=None,
              caption: str = "", parent_id: str | None = None,
              mp4: str | Path | None = None, privacy: str | None = None,
              auto_train: bool = True) -> dict[str, Any]:
    """The atomic 'experiment' step. Returns a status dict for the CLI/web."""
    candidates = prepare_candidates(episode_id, parent_id=parent_id, niche=niche,
                                    hashtags=hashtags, caption=caption)
    summary = optimize.summarize_population(candidates)
    pub = publish_best(episode_id, candidates, mp4=mp4, privacy=privacy, niche=niche)

    # immediate (t=0) snapshot so there's always at least one data point
    hv.harvest_once(pub["experience_id"])

    if auto_train:
        tr.maybe_retrain()

    return {
        "episode_id": episode_id,
        "experience_id": pub["experience_id"],
        "population": summary,
        "chosen_variation": pub["chosen"].get("variation"),
        "chosen_predicted_reward": pub["chosen"].get("_predicted_reward"),
        "platform": learn_config.platform,
        "upload_status": pub["upload"].get("status"),
        "video_id": pub["upload"].get("video_id"),
        "next_harvest": "scheduled per poll_schedule",
    }


def status() -> dict[str, Any]:
    store = db.open_db()
    n = store.count()
    posted = store.list_experiences(only_posted=True)
    chosen = store.list_experiences(only_chosen=True)
    # best performer
    best = None
    best_r = -1.0
    for e in posted:
        if e.reward > best_r:
            best_r = e.reward
            best = e
    engd = eng.get_engine().diagnostics()
    store.close()
    return {
        "total_experiences": n,
        "posted": len(posted),
        "chosen": len(chosen),
        "best_reward": round(best_r, 4) if best else 0.0,
        "best_variation": best.variation if best else None,
        "engine": engd,
        "platform": learn_config.platform,
        "active_reward_preset": learn_config.active_reward_preset,
    }
