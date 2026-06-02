"""Replay a single episode of a trained Intercepterp policy.

Usage:
    python viz/replay.py --run runs/latest
    python viz/replay.py --run runs/20260602_143200 --episode 42
    python viz/replay.py --run runs/latest --save replay.gif --no-show

Produces an animated top-down view (interceptor and intruder paths plus the FOV
cone), a bearing angle theta(t) time series, and a terminal summary line
(SUCCESS/FAIL, reason, t_final, r_final). See spec section 8.

The --episode index is used as the reset seed, so a given index always replays
the same scenario for a given config.
"""
from __future__ import annotations

# Make the repo root importable when run as a script (python viz/replay.py).
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

import numpy as np
import yaml
from sb3_contrib import RecurrentPPO

from envs.intercept_env import ActionConfig, InterceptEnv


def _newest_run() -> pathlib.Path:
    runs_dir = _ROOT / "runs"
    candidates = [d for d in runs_dir.iterdir() if d.is_dir()] if runs_dir.exists() else []
    if not candidates:
        raise FileNotFoundError("no runs/ subdirectories found to resolve 'latest'")
    return max(candidates, key=lambda d: d.stat().st_mtime)


def resolve_run_dir(run_str: str) -> pathlib.Path:
    path = pathlib.Path(run_str)
    if "latest" in path.parts:
        return _newest_run()
    return path


def find_model(run_dir: pathlib.Path, explicit: str | None) -> pathlib.Path:
    if explicit:
        return pathlib.Path(explicit)
    for name in ("best_model.zip", "final_model.zip"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no best_model.zip or final_model.zip in {run_dir}")


def rollout(model, env: InterceptEnv, seed: int, deterministic: bool) -> dict:
    """Run one episode, recording the full trajectory for plotting."""
    obs, info = env.reset(seed=seed)

    rec = {
        "t": [info["t"]],
        "xi": [info["interceptor"]["x"]],
        "yi": [info["interceptor"]["y"]],
        "psi": [info["interceptor"]["psi"]],
        "xt": [info["intruder"]["x"]],
        "yt": [info["intruder"]["y"]],
        "bearing": [info["bearing_true"]],
        "range": [info["range"]],
    }

    lstm_states = None
    episode_start = np.array([True])
    done = False
    while not done:
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_start, deterministic=deterministic
        )
        obs, _, terminated, truncated, info = env.step(action)
        episode_start = np.array([False])
        done = terminated or truncated

        rec["t"].append(info["t"])
        rec["xi"].append(info["interceptor"]["x"])
        rec["yi"].append(info["interceptor"]["y"])
        rec["psi"].append(info["interceptor"]["psi"])
        rec["xt"].append(info["intruder"]["x"])
        rec["yt"].append(info["intruder"]["y"])
        rec["bearing"].append(info["bearing_true"])
        rec["range"].append(info["range"])

    rec["reason"] = info["termination_reason"]
    rec["success"] = info["termination_reason"] == "success"
    rec["t_final"] = float(info["t"])
    rec["r_final"] = float(info["range"])
    for key in ("t", "xi", "yi", "psi", "xt", "yt", "bearing", "range"):
        rec[key] = np.asarray(rec[key])
    return rec


def animate(rec: dict, fov_half_angle_rad: float, fps: int,
            save: str | None, show: bool) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Wedge

    fig, (ax_traj, ax_brg) = plt.subplots(1, 2, figsize=(12, 6))

    # --- Trajectory panel limits with padding. ---
    all_x = np.concatenate([rec["xi"], rec["xt"]])
    all_y = np.concatenate([rec["yi"], rec["yt"]])
    pad = 0.08 * max(np.ptp(all_x), np.ptp(all_y), 1.0) + 20.0
    ax_traj.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax_traj.set_ylim(all_y.min() - pad, all_y.max() + pad)
    ax_traj.set_aspect("equal")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_title("trajectory")

    (line_i,) = ax_traj.plot([], [], "-", color="tab:blue", lw=1.5, label="interceptor")
    (line_t,) = ax_traj.plot([], [], "-", color="tab:red", lw=1.5, label="intruder")
    (pt_i,) = ax_traj.plot([], [], "o", color="tab:blue", ms=6)
    (pt_t,) = ax_traj.plot([], [], "^", color="tab:red", ms=11)
    ax_traj.legend(loc="upper right", fontsize=9)

    # --- Bearing panel. ---
    bearing_deg = np.degrees(rec["bearing"])
    fov_deg = np.degrees(fov_half_angle_rad)
    ax_brg.axhline(fov_deg, color="k", ls="--", lw=0.8, label="FOV edge")
    ax_brg.axhline(-fov_deg, color="k", ls="--", lw=0.8)
    ax_brg.set_xlim(rec["t"][0], max(rec["t"][-1], rec["t"][0] + 1e-3))
    ax_brg.set_ylim(-max(fov_deg * 1.5, np.abs(bearing_deg).max() * 1.1 + 1.0),
                    max(fov_deg * 1.5, np.abs(bearing_deg).max() * 1.1 + 1.0))
    ax_brg.set_xlabel("t [s]")
    ax_brg.set_ylabel("true bearing theta [deg]")
    ax_brg.grid(True, alpha=0.3)
    ax_brg.set_title("bearing to target")
    (line_b,) = ax_brg.plot([], [], "-", color="tab:green", lw=1.5)
    ax_brg.legend(loc="upper right", fontsize=9)

    wedge_holder: list = [None]

    def update(k: int):
        line_i.set_data(rec["xi"][: k + 1], rec["yi"][: k + 1])
        line_t.set_data(rec["xt"][: k + 1], rec["yt"][: k + 1])
        pt_i.set_data([rec["xi"][k]], [rec["yi"][k]])
        pt_t.set_data([rec["xt"][k]], [rec["yt"][k]])

        if wedge_holder[0] is not None:
            wedge_holder[0].remove()
        psi = rec["psi"][k]
        wedge_r = 0.15 * max(rec["range"][k], 1.0) + 40.0
        wedge = Wedge(
            (rec["xi"][k], rec["yi"][k]),
            wedge_r,
            np.degrees(psi - fov_half_angle_rad),
            np.degrees(psi + fov_half_angle_rad),
            alpha=0.15,
            color="tab:blue",
        )
        ax_traj.add_patch(wedge)
        wedge_holder[0] = wedge

        line_b.set_data(rec["t"][: k + 1], bearing_deg[: k + 1])
        ax_traj.set_title(f"trajectory   t={rec['t'][k]:5.2f}s   range={rec['range'][k]:7.1f}m")
        return line_i, line_t, pt_i, pt_t, line_b

    n_frames = len(rec["t"])
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    fig.tight_layout()

    if save:
        save_path = pathlib.Path(save)
        if save_path.suffix.lower() == ".gif":
            anim.save(str(save_path), writer=PillowWriter(fps=fps))
        else:
            anim.save(str(save_path), fps=fps)
        print(f"[replay] saved animation to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main(args: argparse.Namespace) -> None:
    run_dir = resolve_run_dir(args.run)
    model_path = find_model(run_dir, args.model)

    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        cfg_path = _ROOT / "config" / "defaults.yaml"
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    stage = args.stage or len(config["curriculum"]["thresholds"])
    print(f"[replay] run={run_dir}  model={model_path.name}  stage={stage}  episode(seed)={args.episode}")

    model = RecurrentPPO.load(str(model_path), device="cpu")
    env = InterceptEnv(config, ActionConfig(), curriculum_stage=stage, render_mode=None)
    rec = rollout(model, env, seed=args.episode, deterministic=not args.stochastic)
    env.close()

    outcome = "SUCCESS" if rec["success"] else "FAIL"
    print("-" * 56)
    print(f"  {outcome:8s} reason={rec['reason']:9s} "
          f"t_final={rec['t_final']:5.2f}s  r_final={rec['r_final']:7.1f}m")
    print("-" * 56)

    fov_half_angle_rad = np.radians(float(config["physics"]["fov_half_angle"]))
    animate(rec, fov_half_angle_rad, fps=args.fps, save=args.save, show=not args.no_show)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay one Intercepterp episode.")
    p.add_argument("--run", required=True, help="run directory (supports 'latest')")
    p.add_argument("--episode", type=int, default=0, help="reset seed = episode index")
    p.add_argument("--model", default=None, help="explicit model path override")
    p.add_argument("--stage", type=int, default=None, help="curriculum stage (default: final)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--save", default=None, help="save animation to this path (.gif/.mp4)")
    p.add_argument("--no-show", action="store_true", help="do not open a window")
    p.add_argument("--stochastic", action="store_true", help="sample actions")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
