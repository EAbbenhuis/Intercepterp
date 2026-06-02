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
import os
import shutil

import torch
import yaml
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from envs.intercept_env import ActionConfig, InterceptEnv
from training.callbacks import EntropyDecayCallback, build_callbacks
from training.curriculum import CurriculumScheduler

ON_KAGGLE = os.path.exists("/kaggle/working")
BASE_DIR = "/kaggle/working/runs" if ON_KAGGLE else "runs"


def load_config(path: str | pathlib.Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# Config blocks searched, in order, when resolving a bare --override key to its
# dotted location. The first block that holds the key wins, which mirrors the
# single-source-of-truth layout of defaults.yaml.
_OVERRIDE_BLOCKS = ("reward", "training", "physics", "init", "tuning")


def apply_overrides(config: dict, overrides: list[str]) -> None:
    """Apply --override KEY=VALUE pairs to config in place.

    KEY is a leaf parameter name (e.g. closing_reward), resolved to its block by
    the fixed _OVERRIDE_BLOCKS search order. Values are always numeric and cast
    to float; the int-typed consumers re-cast when they read the config, so a
    float value is safe to store.
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--override expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        key = key.strip()
        value = float(raw.strip())
        for block in _OVERRIDE_BLOCKS:
            if block in config and key in config[block]:
                config[block][key] = value
                print(f"[train] override: {block}.{key} = {value}")
                break
        else:
            raise KeyError(
                f"--override key {key!r} not found in blocks {list(_OVERRIDE_BLOCKS)}"
            )


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

    # Apply any --override KEY=VALUE pairs to the freshly loaded config.
    apply_overrides(config, getattr(args, "override", []))

    # Fast tuning mode: a short diagnostic run with close spawns and the
    # curriculum frozen at the tuning stage. Overrides are applied to the
    # in-memory config so every downstream consumer (envs, hyperparameters)
    # sees one consistent picture. Never a production run.
    tuning = bool(getattr(args, "tuning", False))
    if tuning:
        tune = config["tuning"]
        print("=" * 60)
        print("  TUNING MODE ENABLED - not a production run")
        print(f"  Range: {tune['range_mean']}m, Steps: {tune['total_timesteps']}")
        print("=" * 60)
        config["init"]["range_mean"] = tune["range_mean"]
        config["init"]["range_std"] = tune["range_std"]
        config["training"]["total_timesteps"] = tune["total_timesteps"]

    tcfg = config["training"]
    ccfg = config["curriculum"]
    output_cfg = config.get("output", {})

    # Resolve overridable hyperparameters.
    total_timesteps = args.timesteps or int(tcfg["total_timesteps"])
    n_envs = args.n_envs or int(tcfg["n_envs"])

    # Curriculum stage: tuning locks to the tuning stage; otherwise the CLI stage.
    stage = int(config["tuning"]["stage"]) if tuning else args.stage

    # Run directory.
    if args.run_dir:
        run_dir = pathlib.Path(args.run_dir)
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if ON_KAGGLE:
            base_dir = pathlib.Path(BASE_DIR)
        else:
            base_dir = pathlib.Path(output_cfg.get("base_dir", BASE_DIR))
        if not base_dir.is_absolute():
            base_dir = _ROOT / base_dir
        run_dir = base_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot the exact config used (single source of truth for the run).
    shutil.copyfile(args.config, run_dir / "config.yaml")

    print(f"[train] run_dir   = {run_dir}")
    print(f"[train] timesteps = {total_timesteps}")
    print(f"[train] n_envs    = {n_envs}  (subproc={args.subproc})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Environments.
    train_env = build_vec_env(config, n_envs, stage, args.seed, args.subproc)
    eval_env = build_vec_env(config, 1, stage, args.seed + 10_000, False)

    # Curriculum + callbacks. In tuning mode the curriculum is frozen: a single
    # stage scheduler is final from the start, so update() never advances and the
    # env stays at the tuning stage for the whole diagnostic run. The thresholds
    # in config are untouched; this only affects the in-memory tuning scheduler.
    freeze_curriculum = tuning and bool(config["tuning"].get("freeze_curriculum", True))
    if freeze_curriculum:
        scheduler = CurriculumScheduler(
            thresholds=[None],
            min_success_rate=None,
            window_size=int(ccfg["window_size"]),
            patience=int(ccfg["patience"]),
        )
    else:
        scheduler = CurriculumScheduler(
            thresholds=ccfg["thresholds"],
            min_success_rate=ccfg.get("min_success_rate", 0.70),
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
        # RecurrentPPO does not accept a callable schedule for ent_coef (it would
        # break at the first update with function * Tensor). Pass a fixed float
        # and decay it at runtime via EntropyDecayCallback below.
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(run_dir / "tensorboard"),
        seed=args.seed,
        device=device,
        verbose=1,
    )

    # Entropy decay: linearly anneal ent_coef 0.02 -> 0.001 over the first 60% of
    # training, then hold. total_timesteps is the resolved value (tuning or
    # production), so the decay stays proportional in both modes.
    entropy_cb = EntropyDecayCallback(
        initial_value=0.02,
        final_value=0.001,
        end_fraction=0.6,
        total_timesteps=total_timesteps,
    )

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[callbacks, entropy_cb],
            progress_bar=False,
        )
    finally:
        # Always persist the final model, even on KeyboardInterrupt.
        final_path = run_dir / "final_model.zip"
        model.save(final_path)
        print(f"[train] saved final model to {final_path}")
        train_env.close()
        eval_env.close()

        best_path = run_dir / "best_model.zip"
        print(f"[train] best model path: {best_path}")

        # Evaluate on the stage that was actually trained: tuning freezes the
        # scheduler at stage 1, otherwise this is the current curriculum stage.
        eval_stage = scheduler.current_stage
        print(f"[train] done. evaluate with:\n"
          f"  python eval/eval.py --model {run_dir / 'best_model.zip'} "
          f"--n-eps 100 --stage {eval_stage}")
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
    p.add_argument("--eval-freq", type=int, default=50_000,
                   help="timesteps between evaluations (converted to per-env calls)")
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--subproc", action="store_true",
                   help="use SubprocVecEnv instead of DummyVecEnv")
    p.add_argument(
        "--tuning",
        action="store_true",
        help="Enable fast tuning mode: 300k steps, 300m range, stage 1 only."
    )
    p.add_argument(
        "--override",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help="Override a config value. Can be used multiple times."
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
