"""Interceptor agent: fixed-wing pursuer with Dubins kinematics.

The interceptor flies at constant speed and is steered by a single turn-rate
command. Action unpacking is index-based so that a thrust channel can be added
later by reading action[1] and adding one line in apply_action (see CLAUDE.md).

Internal state is SI throughout. The keyword-only arguments default to the spec
section 4.4 values but are normally injected from config by InterceptEnv.
"""
from __future__ import annotations

import numpy as np

from envs.base_env import wrap_to_pi

# Spec section 4.4 / section 7 defaults, in SI for the constructor defaults.
_DEFAULT_SPEED = 78.0                 # m/s
_DEFAULT_PSI_DOT_MAX = np.radians(45.0)  # rad/s  (45 deg/s)
_DEFAULT_DT = 0.05                    # s
_DEFAULT_RANGE_MEAN = 1000.0          # m
_DEFAULT_RANGE_STD = 100.0            # m
_DEFAULT_BEARING_OFFSET_MAX = np.radians(5.0)  # rad  (5 deg)


class Interceptor:
    """Constant-speed Dubins vehicle steered by a normalized turn-rate action."""

    def __init__(
        self,
        *,
        speed: float = _DEFAULT_SPEED,
        psi_dot_max: float = _DEFAULT_PSI_DOT_MAX,
        dt: float = _DEFAULT_DT,
        range_mean: float = _DEFAULT_RANGE_MEAN,
        range_std: float = _DEFAULT_RANGE_STD,
        bearing_offset_max: float = _DEFAULT_BEARING_OFFSET_MAX,
    ) -> None:
        """
        Args:
            speed:              forward speed, m/s (fixed in v1).
            psi_dot_max:        maximum turn rate magnitude, rad/s.
            dt:                 integration timestep, s (stored so apply_action
                                matches the spec signature, which takes no dt).
            range_mean:         spawn range mean, m.
            range_std:          spawn range half-width, m (see reset()).
            bearing_offset_max: spawn bearing offset half-width, rad.
        """
        self.speed = float(speed)
        self.psi_dot_max = float(psi_dot_max)
        self.dt = float(dt)
        self.range_mean = float(range_mean)
        self.range_std = float(range_std)
        self.bearing_offset_max = float(bearing_offset_max)

        # State, set in reset().
        self.x = 0.0
        self.y = 0.0
        self.psi = 0.0

    def reset(self, intruder_state: dict, rng: np.random.Generator) -> None:
        """Place the interceptor relative to the (already reset) intruder.

        Geometry (spec section 4.4):
          - range R from the intruder, drawn uniformly. Spec 4.4 states
            R ~ U(900, 1100) m; section 7 names the config keys range_mean and
            range_std. We follow the explicit distributional statement in 4.4
            and draw R ~ U(range_mean - range_std, range_mean + range_std),
            which equals U(900, 1100) for the default config. (Ambiguity flagged:
            "range_std" reads like a Gaussian sigma, but 4.4 specifies a uniform
            band; the uniform band is bounded and matches 4.4's literal numbers.)
          - position offset by a small angle U(-offset_max, +offset_max) from the
            intruder's heading, so the interceptor sits ahead of the intruder
            (head-on geometry).
          - own heading pointed exactly at the intruder's initial position. The
            spec says "roughly toward"; pointing exactly at it is the cleanest
            reading and guarantees the target is inside the FOV at spawn, so no
            episode terminates at t = 0. (Ambiguity flagged: "roughly".)
        """
        r = rng.uniform(
            self.range_mean - self.range_std,
            self.range_mean + self.range_std,
        )
        offset = rng.uniform(-self.bearing_offset_max, self.bearing_offset_max)

        # Direction from the intruder out to the interceptor.
        placement_dir = intruder_state["psi"] + offset
        self.x = intruder_state["x"] + r * np.cos(placement_dir)
        self.y = intruder_state["y"] + r * np.sin(placement_dir)

        # Heading: point straight back at the intruder's initial position.
        dx = intruder_state["x"] - self.x
        dy = intruder_state["y"] - self.y
        self.psi = wrap_to_pi(np.arctan2(dy, dx))

    def apply_action(self, action: np.ndarray) -> None:
        """Integrate one timestep from a normalized action.

        action[0] is the turn-rate command in [-1, 1]. The env keeps the action
        space at [-1, 1] always (hard rule in CLAUDE.md); here we clip to that
        range defensively, then scale to physical rad/s by psi_dot_max. This is
        the reconciliation of the [-1, 1] contract with the spec 4.4 pseudocode,
        which folded the clip-and-scale into a single clip-to-psi_dot_max line.

        To add thrust later: read action[1], scale to m/s^2, and update
        self.speed. That is the single extra line referenced in CLAUDE.md.
        """
        a0 = float(np.clip(action[0], -1.0, 1.0))
        psi_dot_cmd = a0 * self.psi_dot_max          # rad/s
        self.psi = wrap_to_pi(self.psi + psi_dot_cmd * self.dt)
        self.x += self.speed * np.cos(self.psi) * self.dt
        self.y += self.speed * np.sin(self.psi) * self.dt
        # When thrust is added: self.speed += a1_scaled * self.dt

    def get_state(self) -> dict:
        """Return the kinematic state as a plain dict (SI units)."""
        return {"x": self.x, "y": self.y, "psi": self.psi, "speed": self.speed}
