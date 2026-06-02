"""Evaluation metrics for Intercepterp.

Reduces a list of per-episode summaries into the headline metrics written to
eval_results.json (spec section 8):

    intercept_rate, mean_time_to_intercept, mean_bearing_error,
    fov_loss_rate, timeout_rate, stage

Each per-episode summary is a dict with keys:
    success      bool   detonation within blast radius
    reason       str    "success" | "fov_loss" | "timeout"
    t_final      float  episode end time, s
    mean_bearing float  mean |bearing_true| over the episode, rad

All angles stay in radians (SI). mean_bearing_error is the mean over episodes of
each episode's mean absolute true bearing, i.e. how well the nose tracked the
target on average.
"""
from __future__ import annotations

from collections.abc import Sequence


def _rate(items: Sequence[bool]) -> float:
    return float(sum(items) / len(items)) if items else 0.0


def compute_metrics(episodes: Sequence[dict], stage: int) -> dict:
    """Aggregate per-episode summaries into the eval_results.json payload.

    Args:
        episodes: list of per-episode summary dicts (see module docstring).
        stage:    curriculum stage the evaluation was run at.

    Returns:
        dict with the spec section 8 keys, plus n_episodes for context.
    """
    n = len(episodes)
    if n == 0:
        return {
            "intercept_rate": 0.0,
            "mean_time_to_intercept": None,
            "mean_bearing_error": None,
            "fov_loss_rate": 0.0,
            "timeout_rate": 0.0,
            "stage": int(stage),
            "n_episodes": 0,
        }

    successes = [e for e in episodes if e["success"]]
    ttis = [e["t_final"] for e in successes]
    bearings = [e["mean_bearing"] for e in episodes if e.get("mean_bearing") is not None]

    return {
        "intercept_rate": _rate([e["success"] for e in episodes]),
        # None when nothing was intercepted: a mean over an empty set is undefined.
        "mean_time_to_intercept": (sum(ttis) / len(ttis)) if ttis else None,
        "mean_bearing_error": (sum(bearings) / len(bearings)) if bearings else None,
        "fov_loss_rate": _rate([e["reason"] == "fov_loss" for e in episodes]),
        "timeout_rate": _rate([e["reason"] == "timeout" for e in episodes]),
        "stage": int(stage),
        "n_episodes": n,
    }
