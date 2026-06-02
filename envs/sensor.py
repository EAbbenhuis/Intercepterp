"""Sensor model: the exact interface between simulation truth and policy input.

This is the block replaced by a YOLO + range-estimation pipeline when the system
moves into Simulink (spec section 9, sensor slot). It maps the true relative
geometry to a 3-dim noisy observation and a field-of-view gate. Keeping this
mapping isolated means the policy never sees ground truth.

Observation vector (spec section 6), shape (3,), float32:
    [0] theta_obs  bearing to target, rad, true + N(0, sigma_theta)
    [1] r_obs      range to target, m, true * (1 + N(0, range_noise_frac))
    [2] phi_self   own heading, rad, noiseless

All internal values are SI. Noise std and FOV half-angle are stored in radians;
the env converts the config (degrees) before construction.
"""
from __future__ import annotations

import numpy as np

from envs.base_env import wrap_to_pi

# Spec section 4.5 / section 7 defaults, in SI for the constructor defaults.
_DEFAULT_BEARING_NOISE_STD = np.radians(0.5)  # rad  (0.5 deg)
_DEFAULT_RANGE_NOISE_FRAC = 0.05              # fraction of true range
_DEFAULT_FOV_HALF_ANGLE = np.radians(30.0)    # rad  (30 deg)


class SensorModel:
    """Bearing + noisy range sensor with a hard field-of-view gate."""

    def __init__(
        self,
        *,
        bearing_noise_std: float = _DEFAULT_BEARING_NOISE_STD,
        range_noise_frac: float = _DEFAULT_RANGE_NOISE_FRAC,
        fov_half_angle: float = _DEFAULT_FOV_HALF_ANGLE,
    ) -> None:
        """
        Args:
            bearing_noise_std: additive bearing noise std, rad.
            range_noise_frac:  multiplicative range noise fraction (unitless).
            fov_half_angle:    field-of-view half-angle, rad.
        """
        self.bearing_noise_std = float(bearing_noise_std)
        self.range_noise_frac = float(range_noise_frac)
        self.fov_half_angle = float(fov_half_angle)

        # Cached outputs of the most recent observe() call.
        # last_bearing is the NOISY observed bearing (theta_obs); the reward uses
        # it directly, per spec section 5 (r_step = -0.1 * |theta_obs|).
        self.last_bearing = 0.0
        # True quantities are exposed for the env's termination logic, which must
        # act on real geometry rather than the noisy estimate.
        self.last_true_bearing = 0.0
        self.last_true_range = 0.0
        self.last_in_fov = True

    def observe(
        self,
        interceptor_state: dict,
        intruder_state: dict,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, bool]:
        """Produce the noisy observation and the FOV gate.

        Returns:
            obs:    np.ndarray shape (3,), float32 = [theta_obs, r_obs, phi_self]
            in_fov: bool. False triggers episode termination (FOV loss).

        The FOV gate uses the TRUE bearing, not the noisy one: a monocular camera
        physically produces a detection only when the target lies within the
        frame, and the bearing noise is the error on that detection. Gating on
        truth also keeps termination from firing spuriously on sensor noise.
        (Ambiguity flagged: the spec writes "|theta| < FOV/2" without stating
        true vs noisy; true is the physically coherent, conservative reading.)
        """
        true_bearing = self._bearing(interceptor_state, intruder_state)
        true_range = self._range(interceptor_state, intruder_state)

        # FOV gate on truth. Boundary inclusive: loss only when |theta| > FOV/2.
        in_fov = bool(abs(true_bearing) <= self.fov_half_angle)

        # Noisy observation.
        theta_obs = wrap_to_pi(true_bearing + rng.normal(0.0, self.bearing_noise_std))
        r_obs = true_range * (1.0 + rng.normal(0.0, self.range_noise_frac))
        r_obs = max(0.0, r_obs)  # range is non-negative; env also clips to bounds
        phi_self = wrap_to_pi(interceptor_state["psi"])

        # Cache for the env (reward and info).
        self.last_bearing = float(theta_obs)
        self.last_true_bearing = float(true_bearing)
        self.last_true_range = float(true_range)
        self.last_in_fov = in_fov

        obs = np.array([theta_obs, r_obs, phi_self], dtype=np.float32)
        return obs, in_fov

    def _bearing(self, interceptor: dict, intruder: dict) -> float:
        """Angle from the interceptor body x-axis to the intruder, in [-pi, pi]."""
        dx = intruder["x"] - interceptor["x"]
        dy = intruder["y"] - interceptor["y"]
        bearing_world = np.arctan2(dy, dx)
        return wrap_to_pi(bearing_world - interceptor["psi"])

    def _range(self, interceptor: dict, intruder: dict) -> float:
        """Euclidean range between interceptor and intruder, in metres."""
        dx = intruder["x"] - interceptor["x"]
        dy = intruder["y"] - interceptor["y"]
        return float(np.sqrt(dx**2 + dy**2))
