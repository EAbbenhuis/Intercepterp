"""Live training dashboard for Intercepterp.

Polls the dashboard.json that training/callbacks.py writes (every 10 episodes)
and refreshes a single matplotlib window with four panels (spec section 8):

  - rolling intercept rate vs episode (primary metric), stage changes marked
  - mean |theta| per episode, in degrees (secondary metric)
  - time-to-intercept histogram (refreshes as new intercepts arrive)
  - a status panel with the current curriculum stage, highlighted when it just
    advanced

Non-blocking via plt.pause(), no web server. Run it in a second terminal
alongside training:

    python viz/dashboard.py --run runs/latest
"""
from __future__ import annotations

# Make the repo root importable when run as a script (python viz/dashboard.py).
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import time


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


def read_dashboard(path: pathlib.Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # Mid-write or transient read error; skip this tick.
        return None


def run(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    run_dir = resolve_run_dir(args.run)
    json_path = run_dir / "dashboard.json"
    print(f"[dashboard] watching {json_path} (Ctrl+C to stop)")

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Intercepterp training dashboard", fontsize=13, fontweight="bold")
    ax_rate, ax_brg = axes[0]
    ax_hist, ax_status = axes[1]

    # History accumulated across polls (the JSON only carries the latest value).
    ep_hist: list[int] = []
    rate_hist: list[float] = []
    brg_hist: list[float] = []
    last_episode = -1
    last_stage = None
    highlight_ticks = 0

    try:
        while plt.fignum_exists(fig.number):
            data = read_dashboard(json_path)
            if data is not None and data["episode"] != last_episode:
                last_episode = data["episode"]
                ep_hist.append(data["episode"])
                rate_hist.append(data["intercept_rate"])
                if data.get("mean_bearing_deg") is not None:
                    brg_hist.append(data["mean_bearing_deg"])
                else:
                    brg_hist.append(float("nan"))

                if last_stage is not None and data["stage"] > last_stage:
                    highlight_ticks = 5  # flash the status panel on advancement
                last_stage = data["stage"]

                _redraw(ax_rate, ax_brg, ax_hist, ax_status, data,
                        ep_hist, rate_hist, brg_hist, highlight_ticks > 0)
                highlight_ticks = max(0, highlight_ticks - 1)
            elif data is None:
                ax_status.clear()
                ax_status.axis("off")
                ax_status.text(0.5, 0.5, "waiting for training...",
                               ha="center", va="center", fontsize=12)

            plt.pause(args.interval)
    except KeyboardInterrupt:
        print("\n[dashboard] stopped.")
    finally:
        plt.ioff()


def _redraw(ax_rate, ax_brg, ax_hist, ax_status, data,
            ep_hist, rate_hist, brg_hist, highlight: bool) -> None:
    stage_changes = data.get("stage_changes", [])

    # Panel 1: rolling intercept rate.
    ax_rate.clear()
    ax_rate.plot(ep_hist, rate_hist, "-", color="tab:blue", lw=1.8)
    for sc in stage_changes:
        ax_rate.axvline(sc, color="tab:orange", ls="--", lw=1.0, alpha=0.8)
    ax_rate.set_ylim(-0.02, 1.02)
    ax_rate.set_xlabel("episode")
    ax_rate.set_ylabel("intercept rate")
    ax_rate.set_title("rolling intercept rate")
    ax_rate.grid(True, alpha=0.3)

    # Panel 2: mean bearing magnitude.
    ax_brg.clear()
    ax_brg.plot(ep_hist, brg_hist, "-", color="tab:green", lw=1.8)
    ax_brg.set_xlabel("episode")
    ax_brg.set_ylabel("mean |theta| [deg]")
    ax_brg.set_title("mean bearing magnitude")
    ax_brg.grid(True, alpha=0.3)

    # Panel 3: time-to-intercept histogram.
    ax_hist.clear()
    ttoi = data.get("ttoi", [])
    if ttoi:
        ax_hist.hist(ttoi, bins=20, color="tab:purple", alpha=0.8)
        ax_hist.set_xlabel("time to intercept [s]")
        ax_hist.set_ylabel("count")
    else:
        ax_hist.text(0.5, 0.5, "no intercepts yet", ha="center", va="center")
    ax_hist.set_title("time-to-intercept (recent)")
    ax_hist.grid(True, alpha=0.3)

    # Panel 4: status.
    ax_status.clear()
    ax_status.axis("off")
    if highlight:
        ax_status.set_facecolor("gold")
        ax_status.patch.set_alpha(0.4)
    brg = data.get("mean_bearing_deg")
    brg_str = f"{brg:.2f} deg" if brg is not None else "n/a"
    lines = [
        f"stage:           {data['stage']}",
        f"episode:         {data['episode']}",
        f"timesteps:       {data.get('timesteps', 0):,}",
        f"intercept rate:  {data['intercept_rate']:.3f}",
        f"mean |theta|:    {brg_str}",
    ]
    ax_status.text(0.05, 0.95, "\n".join(lines), ha="left", va="top",
                   fontsize=13, family="monospace", transform=ax_status.transAxes)
    title = "STAGE ADVANCED" if highlight else "status"
    ax_status.set_title(title, fontweight="bold" if highlight else "normal")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live Intercepterp training dashboard.")
    p.add_argument("--run", required=True, help="run directory (supports 'latest')")
    p.add_argument("--interval", type=float, default=2.0, help="poll interval, s")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
