"""Autonomous Reinforcement-Learning Optimization Engine for kleincannon.

A self-contained subsystem that turns every uploaded video into a training
example: it captures generation metadata, uploads to a platform, harvests
metrics over time, scores outcomes with a configurable reward, and learns
which generation parameters produce high-reward videos via a Contextual Bandit.

Design rules (from the spec):
  * independent subsystem — does NOT touch the generation pipeline.
  * every component sits behind an interface so it can be replaced.
  * no hardcoded objectives; reward weights live in learn.json.
  * no magic numbers — every knob lives in learn.json / config.py.
  * platform-agnostic: only the UploadAdapter + MetricsAdapter change per site.
  * local-first: the loop runs offline against MockTikTok until a real token exists.
"""
from __future__ import annotations

from .config import learn_config, LearnConfig, get_learn_path

__all__ = ["learn_config", "LearnConfig", "get_learn_path"]
