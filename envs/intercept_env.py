"""InterceptEnv: the main Gymnasium environment for Intercepterp.

A 2D bearing-only pursuit-evasion task. The interceptor sees a 3-dim noisy
observation (bearing, range, own heading) and outputs a normalized turn-rate
action in [-1, 1]. The episode ends on detonation (success), FOV loss, or
timeout. See INTERCEPTERP_SPEC.md sections 4.1 and 4.6.

Unit policy: every value stored on this env and passed to the agents/sensor is
SI (m, s, rad). The only place degrees appear is the config dict, which this
constructor converts to radians once, up front.
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium
import numpy as np

from envs.agents.interceptor import Interceptor
from envs.agents.intruder import Intruder
from envs.base_env import BaseEnv, wrap_to_pi
from envs.sensor import SensorModel

# Upper bound of the reported range observation, metres. Generous relative to
# the ~1 km spawn range so the noisy estimate never saturates in normal play.
_OBS_RANGE_MAX = 5000.0


@dataclass
class ActionConfig:
    """Single source of truth for the action space (spec section 4.1).

    Adding a thrust channel later means flipping/adding one field here and adding
    one line in Interceptor.apply_action. Nothing else needs to change, because
    the env derives action_space purely from `dim`.
    """

    # v1: turn rate only. psi_dot in [-psi_dot_max, +psi_dot_max] after scaling.
    turn_rate: bool = True

    # v2 placeholder: uncomment to enable a thrust channel.
    # thrust: bool = False        # dv in [-a_max, +a_max]  m/s^2

    @property
    def dim(self) -> int:
        return sum([self.turn_rate])  # extend: + self.thrust


class InterceptEnv(BaseEnv):
    """Bearing-only fixed-wing interception environment."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: dict,
        action_config: ActionConfig,
        curriculum_stage: int = 1,
        render_mode: str | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.action_config = action_config
        self.curriculum_stage = int(curriculum_stage)
        self.render_mode = render_mode

        # Seed applied once on the first reset if reset() is called without one.
        self._initial_seed = rng_seed
        self._seeded = False

        # --- Unpack config into SI scalars (degrees -> radians here only). ---
        phys = config["physics"]
        self.dt = float(phys["dt"])
        self.blast_radius = float(phys["blast_radius"])
        self.episode_timeout = float(phys["episode_timeout"])
        self.fov_half_angle = np.radians(float(phys["fov_half_angle"]))

        rew = config["reward"]
        self.alpha = float(rew["bearing_penalty"])
        self.reward_success = float(rew["success"])
        self.reward_fov_loss = float(rew["fov_loss"])
        self.reward_timeout = float(rew["timeout"])

        # --- Build agents and sensor with injected, converted parameters. ---
        intr = config["intruder"]
        self.intruder = Intruder(
            self.curriculum_stage,
            speed=float(phys["intruder_speed"]),
            psi_dot_max=np.radians(float(intr["psi_dot_max"])),
            sin_period=float(intr["sin_period"]),
        )

        init = config["init"]
        self.interceptor = Interceptor(
            speed=float(phys["interceptor_speed"]),
            psi_dot_max=np.radians(float(phys["psi_dot_max"])),
            dt=self.dt,
            range_mean=float(init["range_mean"]),
            range_std=float(init["range_std"]),
            bearing_offset_max=np.radians(float(init["bearing_offset_max"])),
        )

        sens = config["sensor"]
        self.sensor = SensorModel(
            bearing_noise_std=np.radians(float(sens["bearing_noise_std"])),
            range_noise_frac=float(sens["range_noise_frac"]),
            fov_half_angle=self.fov_half_angle,
        )

        # --- Spaces. ---
        # Observation: [theta_obs in [-pi, pi], r_obs in [0, RANGE_MAX],
        #               phi_self in [-pi, pi]], float32.
        self.observation_space = gymnasium.spaces.Box(
            low=np.array([-np.pi, 0.0, -np.pi], dtype=np.float32),
            high=np.array([np.pi, _OBS_RANGE_MAX, np.pi], dtype=np.float32),
            dtype=np.float32,
        )
        # Action: always [-1, 1], shape (action_config.dim,). The env scales to
        # physical units inside Interceptor.apply_action.
        self.action_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_config.dim,),
            dtype=np.float32,
        )

        # --- Episode state, populated in reset(). ---
        self.t = 0.0
        self.range = float("inf")
        self.in_fov = True
        self.termination_reason = ""
        self._last_obs = np.zeros(3, dtype=np.float32)
        # Observed (noisy) range at the previous and current step, metres. Used
        # by the closing reward so the agent is paid for shrinking the estimated
        # range. Both initialised in reset() to the initial observed range.
        self.prev_r_obs = 0.0
        self.current_r_obs = 0.0

        # Trajectory buffers for rendering. Only populated when rendering, to
        # keep training (render_mode=None) free of this overhead.
        self._traj_interceptor: list[tuple[float, float]] = []
        self._traj_intruder: list[tuple[float, float]] = []
        self._fig = None
        self._ax = None

    # ------------------------------------------------------------------ API --

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        # Apply the constructor seed once if no explicit seed is given.
        if seed is None and not self._seeded:
            seed = self._initial_seed
        super().reset(seed=seed)
        self._seeded = True

        # Apply the (possibly updated) curriculum stage to the intruder.
        self.intruder.stage = int(self.curriculum_stage)

        # Place agents. self.np_random is the np.random.Generator seeded above.
        self.intruder.reset(self.np_random)
        self.interceptor.reset(self.intruder.get_state(), self.np_random)

        self.t = 0.0
        self.termination_reason = ""

        # Initial observation. The interceptor points at the intruder, so the
        # target starts inside the FOV.
        obs, in_fov = self.sensor.observe(
            self.interceptor.get_state(),
            self.intruder.get_state(),
            self.np_random,
        )
        self.range = self.sensor.last_true_range
        self.in_fov = in_fov
        self._last_obs = self._clip_obs(obs)

        # Seed the closing-reward baseline with the initial observed range, so the
        # first step's closing reward measures the change from t = 0.
        self.prev_r_obs = float(obs[1])
        self.current_r_obs = float(obs[1])

        if self.render_mode == "rgb_array":
            self._traj_interceptor = [(self.interceptor.x, self.interceptor.y)]
            self._traj_intruder = [(self.intruder.x, self.intruder.y)]

        return self._get_obs(), self._get_info()

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        # 1. Move the interceptor (action is normalized; scaled internally).
        self.interceptor.apply_action(action)
        # 2. Move the intruder. Stage 3 reacts to the interceptor's new state.
        self.intruder.step(self.dt, interceptor_state=self.interceptor.get_state())
        # advance time before observing/rewarding so the timeout check is current
        self.t += self.dt
        # 3. Observe through the sensor.
        obs, in_fov = self.sensor.observe(
            self.interceptor.get_state(),
            self.intruder.get_state(),
            self.np_random,
        )
        self.range = self.sensor.last_true_range
        self.in_fov = in_fov
        self._last_obs = self._clip_obs(obs)

        # Roll the observed-range window forward before rewarding: prev_r_obs is
        # last step's observed range, current_r_obs is this step's.
        self.prev_r_obs = self.current_r_obs
        self.current_r_obs = float(obs[1])

        # 4 + 5. Reward and termination.
        reward, terminated, truncated = self._compute_reward()

        if self.render_mode == "rgb_array":
            self._traj_interceptor.append((self.interceptor.x, self.interceptor.y))
            self._traj_intruder.append((self.intruder.x, self.intruder.y))

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------- internals --

    def _clip_obs(self, obs: np.ndarray) -> np.ndarray:
        """Clip an observation into the declared observation_space bounds."""
        return np.clip(
            obs, self.observation_space.low, self.observation_space.high
        ).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        """Return the most recent observation (built in reset/step)."""
        return self._last_obs

    def _get_info(self) -> dict:
        """Auxiliary diagnostics. Not used by the policy; consumed by eval/viz.

        is_success follows the SB3 convention so EvalCallback can log it.
        """
        return {
            "t": self.t,
            "range": self.range,
            "bearing_obs": self.sensor.last_bearing,
            "bearing_true": self.sensor.last_true_bearing,
            "in_fov": self.in_fov,
            "stage": self.curriculum_stage,
            "termination_reason": self.termination_reason,
            "is_success": self.termination_reason == "success",
            "interceptor": self.interceptor.get_state(),
            "intruder": self.intruder.get_state(),
        }

    def _compute_reward(self) -> tuple[float, bool, bool]:
        """Return (reward, terminated, truncated) for the current state.

        Reward v2 (loitering fix). The per-step penalty uses the NOISY observed
        bearing (sensor.last_bearing == theta_obs), per spec section 5, with a
        halved coefficient to weaken the loitering incentive. A closing reward
        pays for shrinking the observed range, so absorbing the (now hardened)
        timeout penalty by orbiting at fixed range is no longer an equilibrium.
        Priority is success, then FOV loss, then timeout: detonation always wins.
        The termination_reason assignments are retained because eval, the info
        dict, and the curriculum's is_success flag all read them.
        """
        # Bearing penalty (halved)
        r_step = -self.config["reward"]["bearing_penalty"] * abs(self.sensor.last_bearing)

        # Closing reward: only reward range decrease, never penalise increase
        # Uses noisy r_obs consistent with GPS-denied sensor model
        closing = max(0.0, self.prev_r_obs - self.current_r_obs)
        r_step += self.config["reward"]["closing_reward"] * closing

        # Terminal conditions
        if self.range < self.config["physics"]["blast_radius"]:
            self.termination_reason = "success"
            return r_step + self.config["reward"]["success"], True, False

        if not self.in_fov:
            self.termination_reason = "fov_loss"
            return r_step + self.config["reward"]["fov_loss"], True, False

        if self.t >= self.config["physics"]["episode_timeout"]:
            self.termination_reason = "timeout"
            return r_step + self.config["reward"]["timeout"], False, True

        self.termination_reason = ""
        return r_step, False, False

    # ---------------------------------------------------------------- render --

    def render(self) -> np.ndarray | None:
        """Top-down RGB frame: paths, interceptor heading arrow, intruder
        triangle, and the shaded FOV wedge. Used by viz/replay.py only.
        """
        if self.render_mode != "rgb_array":
            return None

        # Imported lazily so training never pays the matplotlib import cost.
        import matplotlib
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.patches import Wedge

        if self._fig is None:
            self._fig = Figure(figsize=(6, 6), dpi=100)
            FigureCanvasAgg(self._fig)
            self._ax = self._fig.add_subplot(111)

        ax = self._ax
        ax.clear()

        int_xy = np.array(self._traj_interceptor) if self._traj_interceptor else None
        tgt_xy = np.array(self._traj_intruder) if self._traj_intruder else None

        if int_xy is not None:
            ax.plot(int_xy[:, 0], int_xy[:, 1], "-", color="tab:blue",
                    lw=1.5, label="interceptor")
        if tgt_xy is not None:
            ax.plot(tgt_xy[:, 0], tgt_xy[:, 1], "-", color="tab:red",
                    lw=1.5, label="intruder")

        ic = self.interceptor
        it = self.intruder

        # FOV wedge from the interceptor along its heading.
        wedge_r = 0.15 * max(self.sensor.last_true_range, 1.0) + 50.0
        wedge = Wedge(
            (ic.x, ic.y),
            wedge_r,
            np.degrees(ic.psi - self.fov_half_angle),
            np.degrees(ic.psi + self.fov_half_angle),
            alpha=0.15,
            color="tab:blue",
        )
        ax.add_patch(wedge)

        # Interceptor marker + heading arrow.
        arrow_len = 0.5 * wedge_r
        ax.plot(ic.x, ic.y, "o", color="tab:blue", ms=6)
        ax.annotate(
            "",
            xy=(ic.x + arrow_len * np.cos(ic.psi), ic.y + arrow_len * np.sin(ic.psi)),
            xytext=(ic.x, ic.y),
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.5),
        )
        # Intruder as a triangle.
        ax.plot(it.x, it.y, "^", color="tab:red", ms=10)

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"t = {self.t:5.2f} s    range = {self.range:7.1f} m")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        self._fig.canvas.draw()
        buf = np.asarray(self._fig.canvas.buffer_rgba())
        return buf[:, :, :3].copy()

    def close(self) -> None:
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
            self._ax = None
