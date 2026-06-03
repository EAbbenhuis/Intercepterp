"""Evaluate a trained Intercepterp policy and write eval_results.json.

Usage:
    python eval/eval.py --model runs/latest/best_model.zip --n-eps 100
    python eval/eval.py --model runs/20260602_143200/best_model.zip --stage 3

Runs N independent episodes with the MLP policy, aggregates with
eval.metrics.compute_metrics, and writes the result next to the model
(spec section 8).
"""
from __future__ import annotations

# Make the repo root importable when run as a script (python eval/eval.py).
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.intercept_env import ActionConfig, InterceptEnv
from eval.metrics import compute_metrics


def _newest_run() -> pathlib.Path:
    runs_dir = _ROOT / "runs"
    candidates = [d for d in runs_dir.iterdir() if d.is_dir()] if runs_dir.exists() else []
    if not candidates:
        raise FileNotFoundError("no runs/ subdirectories found to resolve 'latest'")
    return max(candidates, key=lambda d: d.stat().st_mtime)


def resolve_latest(path_str: str) -> pathlib.Path:
    """Resolve a path containing a 'latest' component to the newest run dir."""
    path = pathlib.Path(path_str)
    if "latest" not in path.parts:
        return path
    idx = path.parts.index("latest")
    suffix = path.parts[idx + 1:]
    return _newest_run().joinpath(*suffix)


def load_config(model_dir: pathlib.Path, explicit: str | None) -> dict:
    """Prefer the run's config snapshot, fall back to the repo defaults."""
    if explicit:
        cfg_path = pathlib.Path(explicit)
    elif (model_dir / "config.yaml").exists():
        cfg_path = model_dir / "config.yaml"
    else:
        cfg_path = _ROOT / "config" / "defaults.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def build_eval_env(config: dict, stage: int, seed: int) -> InterceptEnv:
    """Construct the InterceptEnv used for evaluation at a given curriculum stage.

    A single factory so the stage flows to exactly one place (and so tests can
    assert the stage reaches the env). Kept as a plain gymnasium env, not a
    VecEnv: the rollout below reads the rich per-step info dict, which the
    gymnasium API exposes directly.
    """
    return InterceptEnv(
        config, ActionConfig(), curriculum_stage=stage, rng_seed=seed
    )


def load_obs_normalizer(model_dir: pathlib.Path, config: dict):
    """Return an observation-normalising callable from a sibling VecNormalize.

    Training saves vec_normalize.pkl next to the model. If present, its statistics
    are loaded and its normalize_obs is returned so evaluation feeds the policy
    exactly the observation distribution it trained on. Under the project's hard
    rule norm_obs=False, this map is the identity; loading it keeps eval correct
    if observation normalisation is ever enabled. Rewards are never normalised at
    eval time: metrics come from the env's true reward and info, and the loaded
    wrapper has norm_reward=False with training=False so no statistics update.
    Returns the identity map when no vec_normalize.pkl is found.
    """
    vec_normalize_path = model_dir / "vec_normalize.pkl"
    if not vec_normalize_path.exists():
        return lambda obs: obs

    # VecNormalize.load needs a venv to attach to; it is only used as a stateless
    # observation transformer here and never stepped.
    dummy = DummyVecEnv([lambda: InterceptEnv(config, ActionConfig())])
    vec = VecNormalize.load(str(vec_normalize_path), dummy)
    vec.training = False      # do not update stats during eval
    vec.norm_reward = False   # evaluate on true rewards, not normalised
    print(f"[eval] loaded VecNormalize stats from {vec_normalize_path}")
    return vec.normalize_obs


def run_episode(
    model,
    env: InterceptEnv,
    deterministic: bool,
    normalize_obs=lambda obs: obs,
) -> dict:
    """Roll out one episode and return its summary dict for metrics."""
    obs, info = env.reset()
    bearing_sum = 0.0
    steps = 0
    done = False

    while not done:
        action, _ = model.predict(
            normalize_obs(obs),
            deterministic=deterministic,
        )
        obs, _, terminated, truncated, info = env.step(action)
        bearing_sum += abs(float(info["bearing_true"]))
        steps += 1
        done = terminated or truncated

    return {
        "success": info["termination_reason"] == "success",
        "reason": info["termination_reason"],
        "t_final": float(info["t"]),
        "mean_bearing": bearing_sum / steps if steps else None,
    }


def evaluate(args: argparse.Namespace) -> dict:
    model_path = resolve_latest(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    model_dir = model_path.parent

    config = load_config(model_dir, args.config)
    # Evaluation stage defaults to stage 1; override with --stage.
    stage = args.stage

    print(f"[eval] model = {model_path}")
    print(f"[eval] stage = {stage}   n_eps = {args.n_eps}   deterministic = {not args.stochastic}")

    model = PPO.load(str(model_path), device=args.device)
    env = build_eval_env(config, stage, args.seed)
    # If the run saved VecNormalize stats, load them so the policy sees the same
    # observation distribution it trained on (identity under norm_obs=False).
    normalize_obs = load_obs_normalizer(model_dir, config)
    # Seed the first episode; later resets continue the stream for diversity.
    env.reset(seed=args.seed)

    episodes = [
        run_episode(
            model, env, deterministic=not args.stochastic, normalize_obs=normalize_obs
        )
        for _ in range(args.n_eps)
    ]
    env.close()

    results = compute_metrics(episodes, stage=stage)

    out_path = pathlib.Path(args.out) if args.out else model_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    _print_summary(results, out_path)
    return results


def _print_summary(results: dict, out_path: pathlib.Path) -> None:
    tti = results["mean_time_to_intercept"]
    berr = results["mean_bearing_error"]
    print("-" * 44)
    print(f"  intercept_rate          {results['intercept_rate']:.3f}")
    print(f"  mean_time_to_intercept  {tti:.2f} s" if tti is not None else
          "  mean_time_to_intercept  n/a (no intercepts)")
    print(f"  mean_bearing_error      "
          f"{np.degrees(berr):.2f} deg" if berr is not None else
          "  mean_bearing_error      n/a")
    print(f"  fov_loss_rate           {results['fov_loss_rate']:.3f}")
    print(f"  timeout_rate            {results['timeout_rate']:.3f}")
    print(f"  stage                   {results['stage']}")
    print(f"  n_episodes              {results['n_episodes']}")
    print("-" * 44)
    print(f"[eval] wrote {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an Intercepterp policy.")
    p.add_argument("--model", required=True, help="path to a saved .zip model")
    p.add_argument("--n-eps", type=int, default=100, help="number of episodes")
    p.add_argument(
        "--stage",
        type=int,
        default=1,
        help="Curriculum stage to evaluate on. Default: 1."
    )
    p.add_argument("--config", default=None,
                   help="config override (default: run snapshot, then repo defaults)")
    p.add_argument("--out", default=None,
                   help="output JSON path (default: <model_dir>/eval_results.json)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of acting deterministically")
    return p.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())
