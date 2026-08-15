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

import json
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


# ---------------------------------------------------------------------------
# Selection + publish-package emission (manual-upload path)
# ---------------------------------------------------------------------------
def select_and_package(episode_id: str, candidates: list[dict[str, Any]],
                       *, mp4: str | Path | None = None,
                       privacy: str | None = None,
                       niche: str = "") -> dict[str, Any]:
    """Choose the best candidate via the learning engine and emit a publish
    package you upload by hand.

    Returns the package dict and (as a side effect) writes it to
    <learn_dir>/pending/<experience_id>.json with the mp4 copied alongside, so
    the CLI/web can show 'what to upload next' and you never lose it.
    """
    engine = eng.get_engine()
    chosen, mean, std = engine.best(candidates)
    chosen = dict(chosen)

    mp4 = Path(mp4) if mp4 else (Episode.load(episode_id).dir / f"{episode_id}.mp4")
    if not mp4.exists():
        raise FileNotFoundError(f"no mp4 at {mp4}")

    chosen = metadata.embed_posting_time(dict(chosen), time.time())
    chosen["privacy"] = privacy or learn_config.default_privacy
    chosen["title"] = chosen.get("title") or Episode.load(episode_id).topic

    # persist the chosen candidate as a CHOSEN-but-not-posted experience
    store = db.open_db()
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
                            chosen=True,
                            upload_status="ready",   # ready to be uploaded by hand
                            generation_params=chosen)
    store.close()

    pkg = write_publish_package(exp.id, episode_id, mp4, chosen, niche)
    return pkg


def write_publish_package(experience_id: str, episode_id: str, mp4: Path,
                          chosen: dict[str, Any], niche: str = "") -> dict[str, Any]:
    """Serialise a publish package to <learn_dir>/pending/<id>.json and copy the
    mp4 next to it. Returns the package dict (also printed by the CLI)."""
    pending_dir = Path(learn_config.learn_dir) / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    mp4 = Path(mp4)
    dest_mp4 = pending_dir / f"{experience_id}.mp4"
    import shutil
    shutil.copy2(mp4, dest_mp4)

    hashtags = chosen.get("hashtags") or learn_config.default_hashtags
    pkg = {
        "experience_id": experience_id,
        "episode_id": episode_id,
        "mp4": str(dest_mp4),
        "title": chosen.get("title", ""),
        "caption": chosen.get("caption", ""),
        "hashtags": hashtags,
        "post_text": f"{chosen.get('caption', '')} " + " ".join(f"#{h}" for h in hashtags),
        "privacy": chosen.get("privacy", learn_config.default_privacy),
        "variation": chosen.get("variation"),
        "predicted_reward": chosen.get("_predicted_reward"),
        "uncertainty": chosen.get("_uncertainty"),
        "uploaded": False,
    }
    (pending_dir / f"{experience_id}.json").write_text(json.dumps(pkg, indent=2))
    return pkg


def pending_packages() -> list[dict[str, Any]]:
    """All publish packages not yet marked uploaded."""
    pending_dir = Path(learn_config.learn_dir) / "pending"
    if not pending_dir.exists():
        return []
    out = []
    for j in sorted(pending_dir.glob("*.json")):
        try:
            pkg = json.loads(j.read_text())
        except Exception:
            continue
        if not pkg.get("uploaded"):
            out.append(pkg)
    return out


# ---------------------------------------------------------------------------
# Manual metrics recording (the agent reads them from the account, you run this)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# P2 — closed-deal recording (the sparse, true objective)
# ---------------------------------------------------------------------------
def record_deal(*, deal_id: str, value: float, offer: str = "",
                touchpoints: list[dict] | None = None,
                attributed_experience_ids: list[str] | None = None,
                attribution_method: str | None = None,
                confidence: float = 1.0, source_ref: str = "") -> dict[str, Any]:
    """Record a closed deal and attribute its value back to contributing assets.

    This is the entry point the web/CLI call when a deal closes. It persists the
    deal (with provenance via `source_ref`) and credits attributed value to the
    experiences that earned it. Returns a summary for the dashboard.

    Fail-closed: never fabricates a deal; credits only land on real experiences.
    """
    from . import attribution as attr
    res = attr.record_deal(
        deal_id=deal_id, value=value, offer=offer, touchpoints=touchpoints,
        attributed_experience_ids=attributed_experience_ids,
        attribution_method=attribution_method, confidence=confidence,
        source_ref=source_ref)
    # the next retrain will fold the new attributed value into the objective
    tr.maybe_retrain()
    return res


def record_metrics(experience_id: str, metrics: dict[str, float],
                   *, mark_uploaded: bool = True,
                   video_id: str | None = None) -> dict[str, Any]:
    """Record metrics observed on the account (by hand or via the browser-vision
    monitor) for a pending/ready experience.

    Appends an immutable snapshot, scores reward, and flips the experience to
    `uploaded` + `posted_at` (so the scheduled harvester would pick it up if you
    later enable automated re-polls). Returns a summary.
    """
    store = db.open_db()
    exp = store.get_experience(experience_id)
    if exp is None:
        store.close()
        raise KeyError(f"no experience {experience_id}")
    m = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    now = time.time()
    off = 0.0 if exp.posted_at is None else round(now - exp.posted_at, 1)
    snap = db.MetricSnapshot(experience_id=experience_id, t_offset=off,
                             captured_at=now, metrics=m)
    added = store.add_snapshot(snap)
    r = reward.score(m)
    # P2: store the blended objective (engagement prior + any attributed deal
    # value already credited to this experience) so the bandit trains on the
    # true north-star once deals exist, and falls back to engagement otherwise.
    blended = reward.blended_reward(
        m, deal_value=exp.attributed_value, deal_confidence=1.0)
    updates: dict[str, Any] = {"reward": round(r, 5), "blended_reward": round(blended, 5)}
    if mark_uploaded:
        updates["upload_status"] = "uploaded"
        updates["posted_at"] = exp.posted_at or now
    if video_id:
        updates["video_id"] = video_id
    store.update_experience(experience_id, **updates)
    # mark the pending package uploaded so it drops off the 'to publish' list
    pending = Path(learn_config.learn_dir) / "pending" / f"{experience_id}.json"
    if pending.exists():
        try:
            p = json.loads(pending.read_text())
            p["uploaded"] = True
            p["video_id"] = video_id or p.get("video_id")
            pending.write_text(json.dumps(p, indent=2))
        except Exception:
            pass
    store.close()
    return {"experience_id": experience_id, "captured": 1 if added else 0,
            "reward": round(r, 5), "metrics": m, "uploaded": mark_uploaded}


# ---------------------------------------------------------------------------
# Auto-publish (legacy / future automated path — inert while manual_upload=True)
# ---------------------------------------------------------------------------
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

    NOTE: only used when learn_config.manual_upload is False. With manual
    upload (the default), use select_and_package() instead.
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
# One full autonomous cycle: prepare -> (manual package | auto publish) -> seed
# ---------------------------------------------------------------------------
def run_cycle(episode_id: str, *, niche: str = "", hashtags=None,
              caption: str = "", parent_id: str | None = None,
              mp4: str | Path | None = None, privacy: str | None = None,
              auto_train: bool = True) -> dict[str, Any]:
    """The atomic 'experiment' step. Returns a status dict for the CLI/web.

    With manual_upload (default) it emits a publish package instead of
    uploading — you upload the video yourself, then run `kc learn record ...`
    (or let the browser-vision monitor do it) to feed metrics back.
    """
    candidates = prepare_candidates(episode_id, parent_id=parent_id, niche=niche,
                                    hashtags=hashtags, caption=caption)
    summary = optimize.summarize_population(candidates)

    if learn_config.manual_upload:
        pkg = select_and_package(episode_id, candidates, mp4=mp4,
                                 privacy=privacy, niche=niche)
        res: dict[str, Any] = {
            "mode": "manual_upload",
            "experience_id": pkg["experience_id"],
            "population": summary,
            "chosen_variation": pkg["variation"],
            "chosen_predicted_reward": pkg["predicted_reward"],
            "publish_package": pkg,
            "next_step": "upload mp4 by hand, then `kc learn record <id> "
                         "--views .. --likes ..` (or let the monitor do it)",
        }
    else:
        pub = publish_best(episode_id, candidates, mp4=mp4, privacy=privacy, niche=niche)
        # immediate (t=0) snapshot so there's always at least one data point
        hv.harvest_once(pub["experience_id"])
        res = {
            "mode": "auto_upload",
            "experience_id": pub["experience_id"],
            "population": summary,
            "chosen_variation": pub["chosen"].get("variation"),
            "chosen_predicted_reward": pub["chosen"].get("_predicted_reward"),
            "platform": learn_config.platform,
            "upload_status": pub["upload"].get("status"),
            "video_id": pub["upload"].get("video_id"),
            "next_harvest": "scheduled per poll_schedule",
        }

    if auto_train:
        tr.maybe_retrain()

    return res


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
