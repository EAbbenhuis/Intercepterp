"""PPO + LSTM training entry point for Intercepterp.

Trains a RecurrentPPO (sb3-contrib) policy on InterceptEnv with a difficulty
curriculum. The LSTM lets the agent integrate the noisy bearing/range history,
which is what makes a bearing-only intercept tractable.

Usage:
    python training/train.py
    python training/train.py --timesteps 200000 --n-envs 8 --seed 0
    python training/train.py --config config/defaults.yaml --subproc

Every run writes runs/YYYYMMDD_HHMMSS/ with: config.yaml (snapshot),
best_model.zip (EvalCallback), final_model.zip, tensorboard/, and dashboard.json.
"""
from __future__ import annotations

# Make the repo root importable when run as a script (python training/train.py).
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import datetime
import shutil

import yaml
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from envs.intercept_env import ActionConfig, InterceptEnv
from training.callbacks import build_callbacks
from training.curriculum import CurriculumScheduler


def load_config(path: str | pathlib.Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_env_factory(
    config: dict,
    stage: int,
    seed: int,
    rank: int,
    render_mode: str | None = None,
):
    """Return a thunk that builds one Monitor-wrapped InterceptEnv."""

    def _init():
        env = InterceptEnv(
            config,
            ActionConfig(),
            curriculum_stage=stage,
            render_mode=render_mode,
            rng_seed=seed + rank,
        )
        return Monitor(env)

    return _init


def build_vec_env(
    config: dict,
    n_envs: int,
    stage: int,
    seed: int,
    use_subproc: bool,
) -> VecEnv:
    factories = [
        make_env_factory(config, stage, seed, rank) for rank in range(n_envs)
    ]
    if use_subproc and n_envs > 1:
        return SubprocVecEnv(factories)
    return DummyVecEnv(factories)


def train(args: argparse.Namespace) -> pathlib.Path:
    config = load_config(args.config)
    tcfg = config["training"]
    ccfg = config["curriculum"]

    # Resolve overridable hyperparameters.
    total_timesteps = args.timesteps or int(tcfg["total_timesteps"])
    n_envs = args.n_envs or int(tcfg["n_envs"])

    # Run directory.
    if args.run_dir:
        run_dir = pathlib.Path(args.run_dir)
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = _ROOT / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot the exact config used (single source of truth for the run).
    shutil.copyfile(args.config, run_dir / "config.yaml")

    print(f"[train] run_dir   = {run_dir}")
    print(f"[train] timesteps = {total_timesteps}")
    print(f"[train] n_envs    = {n_envs}  (subproc={args.subproc})")
    print(f"[train] device    = {args.device}")

    # Environments.
    train_env = build_vec_env(config, n_envs, args.stage, args.seed, args.subproc)
    eval_env = build_vec_env(config, 1, args.stage, args.seed + 10_000, False)

    # Curriculum + callbacks.
    scheduler = CurriculumScheduler(
        thresholds=ccfg["thresholds"],
        window_size=int(ccfg["window_size"]),
        patience=int(ccfg["patience"]),
    )
    eval_freq_calls = max(args.eval_freq // n_envs, 1)
    callbacks = build_callbacks(
        scheduler,
        eval_env,
        run_dir,
        eval_freq=eval_freq_calls,
        n_eval_episodes=args.eval_episodes,
        dashboard_update_freq=10,
        verbose=1,
    )

    # Model: RecurrentPPO with an LSTM-augmented MLP policy.
    policy_kwargs = dict(
        lstm_hidden_size=int(tcfg["lstm_hidden_size"]),
        n_lstm_layers=int(tcfg["lstm_n_layers"]),
    )
    model = RecurrentPPO(
        "MlpLstmPolicy",
        train_env,
        n_steps=int(tcfg["n_steps"]),
        batch_size=int(tcfg["batch_size"]),
        n_epochs=int(tcfg["n_epochs"]),
        learning_rate=float(tcfg["learning_rate"]),
        clip_range=float(tcfg["clip_range"]),
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(run_dir / "tensorboard"),
        seed=args.seed,
        device=args.device,
        verbose=1,
    )

    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)
    finally:
        # Always persist the final model, even on KeyboardInterrupt.
        final_path = run_dir / "final_model.zip"
        model.save(final_path)
        print(f"[train] saved final model to {final_path}")
        train_env.close()
        eval_env.close()

    print(f"[train] done. evaluate with:\n"
          f"  python eval/eval.py --model {run_dir / 'best_model.zip'} --n-eps 100")
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Intercepterp PPO+LSTM policy.")
    p.add_argument("--config", default=str(_ROOT / "config" / "defaults.yaml"))
    p.add_argument("--timesteps", type=int, default=None,
                   help="override training.total_timesteps")
    p.add_argument("--n-envs", type=int, default=None,
                   help="override training.n_envs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage", type=int, default=1, help="initial curriculum stage")
    p.add_argument("--run-dir", default=None,
                   help="explicit run directory (default runs/<timestamp>)")
    p.add_argument("--device", default="cpu", help="torch device (cpu/cuda)")
    p.add_argument("--eval-freq", type=int, default=50_000,
                   help="timesteps between evaluations (converted to per-env calls)")
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--subproc", action="store_true",
                   help="use SubprocVecEnv instead of DummyVecEnv")
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
