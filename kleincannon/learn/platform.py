"""Platform adapters — the only part of the learning subsystem that knows
about a specific site. Everything upstream (engine, optimizer, harvester) talks
to the `PlatformAdapter` interface, so adding YouTube Shorts / Instagram Reels
/ etc. later means writing one new adapter file, nothing else.

Two implementations ship:
  * `MockTikTok` — a fully functional local simulator. The whole optimization
    loop runs against it offline (no token, no network). Its reward model is a
    *hidden* function of the same features, so the bandit genuinely has
    something to learn. This is what makes the subsystem testable today.
  * `TikTokAdapter` — the real Content Posting API + video query/metrics, with
    OAuth 2.0 + PKCE and resumable chunked upload, exactly as TikTok's
    production endpoints require. It activates automatically once a token file
    exists; until then it raises a clear "needs auth" error.

Security/consent: client key+secret are read from env (never stored in code),
the token is written to learn_dir/tiktok_token.json (git-ignored), and upload
defaults to SELF_ONLY unless the operator explicitly opts into PUBLIC_TO_ALL.
"""
from __future__ import annotations

import json
import os
import secrets
import time
import urllib.parse
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from . import learn_config


@runtime_checkable
class PlatformAdapter(Protocol):
    name: str

    def is_authenticated(self) -> bool: ...
    def auth_url(self, redirect_uri: str, state: str) -> str: ...
    def exchange_code(self, code: str, redirect_uri: str) -> dict: ...
    def upload_video(self, mp4_path: str | Path, meta: dict[str, Any]) -> dict: ...
    def get_metrics(self, video_id: str) -> dict[str, float]: ...


# ---------------------------------------------------------------------------
# Mock platform — local, deterministic-ish simulator with a HIDDEN reward fn.
# ---------------------------------------------------------------------------
class MockTikTok:
    """Offline TikTok stand-in.

    It mints a fake video_id on upload and, on each metrics poll, computes a
    *latent* reward from the same features the engine sees — plus a little
    noise and a time-decay curve so metrics grow then plateau. The bandit
    doesn't know the latent fn; it must learn it from history.
    """

    name = "mock_tiktok"

    def __init__(self, seed: int = 0):
        self._videos: dict[str, dict] = {}
        self._rng = __import__("random").Random(seed)
        self._metrics_log: list[dict] = []

    # --- latent quality function (the "truth" the agent must discover) ---
    def _latent(self, meta: dict[str, Any]) -> float:
        from .features import extract
        f = extract(meta)
        # A deliberately non-trivial hidden surface:
        #  - curiosity + hook help a lot
        #  - controversy helps views but hurts followers
        #  - posting on weekends helps slightly
        #  - very long videos hurt
        s = (2.0 * f.get("content_curiosity", 0)
             + 1.5 * f.get("content_emotion", 0)
             + 1.0 * f.get("content_educational", 0)
             - 1.2 * f.get("content_controversy", 0)
             + 0.8 * f.get("post_wd_Saturday", 0)
             + 0.8 * f.get("post_wd_Sunday", 0)
             - 1.5 * max(0.0, f.get("video_duration_norm", 0) - 0.5)
             + 0.5 * f.get("video_subtitle_density", 0))
        return max(0.0, s)

    def is_authenticated(self) -> bool:
        return True

    def auth_url(self, redirect_uri: str, state: str) -> str:
        return f"mock://auth?state={state}&redirect={urllib.parse.quote(redirect_uri)}"

    def exchange_code(self, code: str, redirect_uri: str) -> dict:
        return {"access_token": f"mock_{secrets.token_hex(8)}",
                "open_id": "mock_open_id", "expires_in": 86400 * 30}


    def upload_video(self, mp4_path: str | Path, meta: dict[str, Any]) -> dict:
        vid = "mock_" + secrets.token_hex(10)
        latent = self._latent(meta)
        self._videos[vid] = {
            "meta": meta, "latent": latent, "posted_at": time.time(),
            "title": meta.get("title", ""), "privacy": meta.get("privacy", "SELF_ONLY"),
        }
        return {"video_id": vid, "status": "uploaded", "mock": True}

    def get_metrics(self, video_id: str, now: float | None = None) -> dict[str, float]:
        v = self._videos.get(video_id)
        if not v:
            return {}
        now = now or time.time()
        t = max(0.0, now - v["posted_at"])
        # growth curve: saturating, scaled by latent quality
        reach = v["latent"] * 4000
        decay = 1.0 - 0.6 * (1.0 / (1.0 + t / 86400.0))   # plateau over ~a day
        noise = self._rng.uniform(0.9, 1.1)
        views = max(0.0, reach * decay * noise * (1.0 - 0.5 / (1.0 + t)))
        ctr = 0.06 + 0.04 * v["latent"]
        likes = views * ctr
        comments = likes * 0.12
        shares = likes * 0.08
        saves = likes * 0.10
        followers = shares * 0.5 + saves * 0.3
        watch_time = views * (2.0 + 3.0 * v["latent"])
        completion = min(0.95, 0.4 + 0.4 * v["latent"])
        profile_visits = views * (0.01 + 0.01 * v["latent"])
        m = {
            "views": round(views, 1), "likes": round(likes, 1),
            "comments": round(comments, 1), "shares": round(shares, 1),
            "saves": round(saves, 1), "followers_gained": round(followers, 1),
            "watch_time": round(watch_time, 1), "completion_rate": round(completion, 3),
            "profile_visits": round(profile_visits, 1),
        }
        self._metrics_log.append({"video_id": video_id, "t": t, "metrics": m})
        return m


# ---------------------------------------------------------------------------
# Real TikTok adapter — production Content Posting API + metrics query.
# Activated automatically when a token file exists.
# ---------------------------------------------------------------------------
class TikTokAdapter:
    """Real TikTok Content Posting API v2 (production endpoints).

    Flow: 1) generate auth URL (PKCE) -> 2) user authorizes in browser ->
    3) exchange `code` for token (cached to learn_dir/tiktok_token.json) ->
    4) upload_video does the create+chunked-upload dance -> 5) get_metrics
    queries video.list / video.query.

    Returns may be null for partner-gated metrics (completion_rate, watch_time,
    profile_visits) — callers must tolerate missing keys.
    """

    name = "tiktok"
    BASE = "https://open.tiktokapis.com/v2"
    AUTH = "https://www.tiktok.com/v2/auth/authorize/"
    TOKEN = "https://open.tiktokapis.com/v2/oauth/token/"

    def __init__(self):
        self.client_key = os.environ.get(learn_config.client_key_env, "")
        self.client_secret = os.environ.get(learn_config.client_secret_env, "")
        self.token_path = Path(learn_config.token_path)
        self._pkce: dict[str, str] = {}

    # ---- auth ----
    def is_authenticated(self) -> bool:
        return self.token_path.exists()

    def _load_token(self) -> dict:
        return json.loads(self.token_path.read_text())

    def auth_url(self, redirect_uri: str, state: str | None = None) -> str:
        if not self.client_key:
            raise RuntimeError("TIKTOK_CLIENT_KEY env not set")
        verifier = secrets.token_urlsafe(64)
        # PKCE: challenge = BASE64URL(SHA256(verifier))
        import base64, hashlib
        chal = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        self._pkce[state or "default"] = verifier
        q = urllib.parse.urlencode({
            "client_key": self.client_key,
            "scope": "video.upload video.list user.info.basic",
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "state": state or secrets.token_hex(8),
            "code_challenge": chal,
            "code_challenge_method": "S256",
        })
        return f"{self.AUTH}?{q}"

    def exchange_code(self, code: str, redirect_uri: str, state: str = "default") -> dict:
        import urllib.request
        verifier = self._pkce.get(state, "")
        data = urllib.parse.urlencode({
            "client_key": self.client_key,
            "client_secret": self.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        }).encode()
        req = urllib.request.Request(self.TOKEN, data=data, method="POST",
                                     headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=20) as r:
            token = json.loads(r.read())
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(json.dumps(token, indent=2))
        return token

    def _bearer(self) -> str:
        return self._load_token()["access_token"]

    def _api(self, path: str, payload: dict) -> dict:
        import urllib.request
        url = self.BASE + path
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     method="POST",
                                     headers={"Authorization": f"Bearer {self._bearer()}",
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    # ---- upload (resumable chunked, per TikTok spec) ----
    def upload_video(self, mp4_path: str | Path, meta: dict[str, Any]) -> dict:
        mp4_path = Path(mp4_path)
        size = mp4_path.stat().st_size
        # 1) initialise upload
        init = self._api("/post/publish/video/init/", {
            "post_info": {
                "title": meta.get("title", "")[:140],
                "description": self._description(meta),
                "privacy_level": meta.get("privacy", learn_config.default_privacy),
                "disable_comment": False,
            },
            "source_info": {"source": "FILE_UPLOAD", "video_size": size,
                            "chunk_size": 5 * 1024 * 1024,
                            "total_chunk_count": max(1, (size + 5 * 1024 * 1024 - 1) // (5 * 1024 * 1024))},
        })
        if init.get("error", {}).get("code") != "ok":
            raise RuntimeError(f"upload init failed: {init}")
        publish_id = init["data"]["publish_id"]
        upload_url = init["data"]["upload_url"]

        # 2) chunked PUT
        chunk = 5 * 1024 * 1024
        with mp4_path.open("rb") as fh:
            idx = 0
            while True:
                data = fh.read(chunk)
                if not data:
                    break
                req = urllib.request.Request(upload_url, data=data, method="PUT",
                                             headers={"Content-Type": "video/mp4",
                                                       "Content-Range": f"bytes {idx}-{idx+len(data)-1}/{size}"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    r.read()
                idx += len(data)

        # 3) finalise
        fin = self._api("/post/publish/video/complete/", {"publish_id": publish_id})
        return {"video_id": publish_id, "status": "submitted",
                "publish_id": publish_id, "response": fin}

    def _description(self, meta: dict[str, Any]) -> str:
        tags = " ".join(f"#{h}" for h in meta.get("hashtags", []))
        return f"{meta.get('caption', '')} {tags}".strip()[:400]

    # ---- metrics ----
    def get_metrics(self, video_id: str) -> dict[str, float]:
        resp = self._api("/video/query/", {"filters": {"video_ids": [video_id]}})
        items = (resp.get("data", {}).get("videos") or [])
        if not items:
            return {}
        v = items[0]
        stats = v.get("stats", {})
        return {
            "views": float(stats.get("video_views") or 0),
            "likes": float(stats.get("likes") or 0),
            "comments": float(stats.get("comments") or 0),
            "shares": float(stats.get("shares") or 0),
            # partner-gated; may be absent
            "completion_rate": float(stats.get("video_completion_rate") or 0.0),
            "followers_gained": float(stats.get("followers_gained") or 0.0),
            "profile_visits": float(stats.get("profile_visits") or 0.0),
            "watch_time": float(stats.get("average_watch_time") or 0.0),
        }


def get_adapter(platform: str | None = None) -> PlatformAdapter:
    """Return the active adapter. Real TikTok if authenticated, else MockTikTok.

    This is the single switch point: upload/harvest/test all call this so the
    loop is identical regardless of platform, and running `kc learn init`
    without a token transparently uses the mock.
    """
    platform = platform or learn_config.platform
    if platform == "tiktok":
        real = TikTokAdapter()
        if real.is_authenticated():
            return real
        return MockTikTok()
    # future: elif platform == "youtube_shorts": return YouTubeAdapter()
    return MockTikTok()
