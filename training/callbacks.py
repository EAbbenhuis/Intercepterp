"""Training callbacks.

Exposes two callbacks (spec: training/callbacks.py contains EvalCallback and
CurriculumCallback):

  - EvalCallback: re-exported from Stable-Baselines3. It evaluates the MLP
    policy used here and saves best_model.zip to the run.
  - CurriculumCallback: feeds episode outcomes to the CurriculumScheduler,
    pushes any stage change into the live training (and eval) envs via set_attr,
    logs curriculum metrics, and writes a small JSON file that the dashboard
    polls.

build_callbacks() assembles the CallbackList that train.py uses.
"""
from __future__ import annotations

import json
import math
import pathlib
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback

from training.curriculum import CurriculumScheduler

__all__ = [
    "EvalCallback",
    "CurriculumCallback",
    "build_callbacks",
]


class CurriculumCallback(BaseCallback):
    """Drives the curriculum and writes the dashboard JSON.

    The dashboard JSON holds the rolling intercept rate (primary metric), the
    rolling mean absolute bearing in degrees (secondary), the current stage, the
    episode count, the episodes at which the stage advanced, and a buffer of
    recent time-to-intercept values for the histogram.
    """

    def __init__(
        self,
        scheduler: CurriculumScheduler,
        run_dir: str | pathlib.Path,
        eval_env=None,
        dashboard_update_freq: int = 10,
        ttoi_buffer: int = 200,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.scheduler = scheduler
        self.run_dir = pathlib.Path(run_dir)
        self.eval_env = eval_env
        self.dashboard_update_freq = int(dashboard_update_freq)
        self._dashboard_path = self.run_dir / "dashboard.json"

        # Episode bookkeeping.
        self._episodes = 0
        self._stage_changes: list[int] = []
        self._mean_bearing_deg: deque[float] = deque(maxlen=scheduler.window_size)
        self._ttoi: deque[float] = deque(maxlen=int(ttoi_buffer))

        # Per-env accumulators for the in-episode mean bearing. Sized lazily.
        self._bearing_sum: list[float] = []
        self._bearing_len: list[int] = []

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self._bearing_sum = [0.0] * n_envs
        self._bearing_len = [0] * n_envs
        # Make sure the envs start on the scheduler's current stage.
        self._set_stage(self.scheduler.current_stage)

    def _set_stage(self, stage: int) -> None:
        # set_wrapper_attr searches the wrapper stack and sets curriculum_stage on
        # the InterceptEnv itself, so the change survives the Monitor wrapper.
        # (Plain VecEnv.set_attr would set it on the Monitor wrapper instead,
        # where InterceptEnv.reset never reads it.)
        self.training_env.env_method("set_wrapper_attr", "curriculum_stage", int(stage))
        if self.eval_env is not None:
            self.eval_env.env_method("set_wrapper_attr", "curriculum_stage", int(stage))

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, (done, info) in enumerate(zip(dones, infos)):
            # Accumulate the true bearing magnitude for this env's episode.
            bearing_true = info.get("bearing_true")
            if bearing_true is not None:
                self._bearing_sum[i] += abs(float(bearing_true))
                self._bearing_len[i] += 1

            if not done:
                continue

            # Episode finished on env i. Finalise its statistics.
            self._episodes += 1
            success = bool(info.get("is_success", False))

            if self._bearing_len[i] > 0:
                self._mean_bearing_deg.append(
                    math.degrees(self._bearing_sum[i] / self._bearing_len[i])
                )
            self._bearing_sum[i] = 0.0
            self._bearing_len[i] = 0

            if success:
                # info["t"] is the time of detonation in seconds.
                self._ttoi.append(float(info.get("t", float("nan"))))

            advanced = self.scheduler.update(success)
            if advanced:
                new_stage = self.scheduler.current_stage
                self._set_stage(new_stage)
                self._stage_changes.append(self._episodes)
                if self.verbose:
                    print(
                        f"[curriculum] advanced to stage {new_stage} "
                        f"at episode {self._episodes} "
                        f"(intercept_rate={self.scheduler.intercept_rate:.3f})"
                    )

            # Log to the SB3 logger so the metrics also reach TensorBoard.
            self.logger.record("curriculum/stage", self.scheduler.current_stage)
            self.logger.record("rollout/intercept_rate", self.scheduler.intercept_rate)
            if self._mean_bearing_deg:
                self.logger.record(
                    "rollout/mean_bearing_deg", float(np.mean(self._mean_bearing_deg))
                )

            if self._episodes % self.dashboard_update_freq == 0:
                self._write_dashboard()

        return True

    def _write_dashboard(self) -> None:
        payload = {
            "episode": self._episodes,
            "stage": self.scheduler.current_stage,
            "intercept_rate": self.scheduler.intercept_rate,
            "mean_bearing_deg": (
                float(np.mean(self._mean_bearing_deg))
                if self._mean_bearing_deg
                else None
            ),
            "ttoi": list(self._ttoi),
            "stage_changes": list(self._stage_changes),
            "timesteps": int(self.num_timesteps),
        }
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._dashboard_path.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(payload, f)
            tmp.replace(self._dashboard_path)  # atomic on the same filesystem
        except OSError:
            # Dashboard output is best-effort; never let it break training.
            pass


def build_callbacks(
    scheduler: CurriculumScheduler,
    eval_env,
    run_dir: str | pathlib.Path,
    *,
    eval_freq: int,
    n_eval_episodes: int = 50,
    dashboard_update_freq: int = 10,
    verbose: int = 1,
) -> CallbackList:
    """Assemble the curriculum + evaluation callback list used by train.py."""
    run_dir = pathlib.Path(run_dir)

    curriculum_cb = CurriculumCallback(
        scheduler=scheduler,
        run_dir=run_dir,
        eval_env=eval_env,
        dashboard_update_freq=dashboard_update_freq,
        verbose=verbose,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=verbose,
    )
    # Curriculum first so a stage change applies before the eval that follows.
    return CallbackList([curriculum_cb, eval_cb])
