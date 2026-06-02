"""Intruder agent: Shahed-136 class target with Dubins kinematics.

Evasion behaviour is selected by the curriculum stage:
  Stage 1: straight flight, no evasion.
  Stage 2: sinusoidal heading weave, period set by config (default 8 s).
  Stage 3: reactive evasion, turns away from the interceptor at full rate.

Internal state is SI throughout (metres, seconds, radians). The keyword-only
physics arguments default to the spec section 4.3 values but are normally
injected from config by InterceptEnv, so nothing is hardcoded at the call site.
"""
from __future__ import annotations

import numpy as np

from envs.base_env import wrap_to_pi

# Spec section 4.3 defaults, expressed in SI for the constructor defaults.
_DEFAULT_SPEED = 50.0                 # m/s
_DEFAULT_PSI_DOT_MAX = np.radians(15.0)  # rad/s  (15 deg/s)
_DEFAULT_SIN_PERIOD = 8.0             # s


class Intruder:
    """Constant-speed Dubins vehicle with stage-dependent evasion."""

    def __init__(
        self,
        stage: int,
        *,
        speed: float = _DEFAULT_SPEED,
        psi_dot_max: float = _DEFAULT_PSI_DOT_MAX,
        sin_period: float = _DEFAULT_SIN_PERIOD,
    ) -> None:
        """
        Args:
            stage:       curriculum stage in {1, 2, 3}. May be reassigned by the
                         env between episodes when the curriculum advances.
            speed:       forward speed, m/s (fixed).
            psi_dot_max: maximum turn rate magnitude, rad/s.
            sin_period:  stage-2 heading-weave period, s.
        """
        self.stage = int(stage)
        self.speed = float(speed)
        self.psi_dot_max = float(psi_dot_max)
        self.sin_period = float(sin_period)

        # State, set in reset().
        self.x = 0.0
        self.y = 0.0
        self.psi = 0.0
        # psi0 is the heading at reset; the stage-2 weave oscillates about it.
        self._psi0 = 0.0
        # Elapsed episode time, needed for the stage-2 sinusoid.
        self._t = 0.0

    def reset(self, rng: np.random.Generator) -> None:
        """Place the intruder at the origin facing a uniformly random heading.

        The interceptor is positioned relative to this state by Interceptor.reset.
        """
        self.x = 0.0
        self.y = 0.0
        self.psi = float(rng.uniform(-np.pi, np.pi))
        self._psi0 = self.psi
        self._t = 0.0

    def step(self, dt: float, interceptor_state: dict | None = None) -> None:
        """Advance one timestep using the current stage's heading law.

        Args:
            dt: timestep, s.
            interceptor_state: required only for stage 3 (reactive evasion);
                ignored otherwise. The env always passes it, which is cheap.
        """
        self._t += dt

        if self.stage <= 1:
            # Stage 1: straight flight. Heading unchanged.
            pass
        elif self.stage == 2:
            # Stage 2: sinusoidal heading weave about the reset heading.
            # The spec fixes the period but not the amplitude. We tie the
            # amplitude to psi_dot_max and the period so the peak turn rate is
            # exactly psi_dot_max: with psi(t) = psi0 + A*sin(2*pi*t/T) the peak
            # rate is A*2*pi/T, so A = psi_dot_max * T / (2*pi). This respects
            # the stated turn-rate bound without introducing a new free
            # parameter. (Ambiguity flagged: amplitude is not given in the spec.)
            amplitude = self.psi_dot_max * self.sin_period / (2.0 * np.pi)
            phase = 2.0 * np.pi * self._t / self.sin_period
            self.psi = wrap_to_pi(self._psi0 + amplitude * np.sin(phase))
        else:
            # Stage 3: reactive evasion. Turn away from the interceptor at the
            # maximum rate. beta is the interceptor's bearing in the intruder
            # body frame; turning opposite its sign pushes the interceptor
            # toward the rear hemisphere.
            if interceptor_state is not None:
                beta = self._bearing_to(interceptor_state)
                # sign(0) == 0 would mean no turn when the interceptor is dead
                # ahead (worst case), so default to a positive turn there.
                direction = -np.sign(beta) if beta != 0.0 else 1.0
                self.psi = wrap_to_pi(self.psi + direction * self.psi_dot_max * dt)
            # If no interceptor_state is supplied, fall back to straight flight.

        # Position integration (forward Euler) with the updated heading.
        self.x += self.speed * np.cos(self.psi) * dt
        self.y += self.speed * np.sin(self.psi) * dt

    def _bearing_to(self, other_state: dict) -> float:
        """Bearing of another agent in this intruder's body frame, in [-pi, pi]."""
        dx = other_state["x"] - self.x
        dy = other_state["y"] - self.y
        return wrap_to_pi(np.arctan2(dy, dx) - self.psi)

    def get_state(self) -> dict:
        """Return the kinematic state as a plain dict (SI units)."""
        return {"x": self.x, "y": self.y, "psi": self.psi, "speed": self.speed}
