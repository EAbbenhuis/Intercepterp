"""Sensor model: the exact interface between simulation truth and policy input.

This is the block replaced by a YOLO + range-estimation pipeline when the system
moves into Simulink (spec section 9, sensor slot). It maps the true relative
geometry to a noisy measurement, runs an Extended Kalman Filter to recover the
relative motion state, and gates on the field of view. Keeping this mapping
isolated means the policy never sees ground truth.

Two observation interfaces are provided:

    observe()       legacy 3-dim noisy observation, kept for backward
                    compatibility:
                        [0] theta_obs  bearing to target, rad, true + N(0, sigma)
                        [1] r_obs      range to target, m, true * (1 + N(0, frac))
                        [2] phi_self   own heading, rad, noiseless

    observe_full()  new 5-dim EKF observation (the current Simulink sensor slot):
                        [0] theta_hat     EKF estimated bearing, rad
                        [1] theta_dot_hat EKF estimated bearing rate, rad/s
                        [2] r_hat         EKF estimated range, m
                        [3] r_dot_hat     EKF estimated range rate, m/s
                        [4] phi_self      own heading, rad, noiseless

The EKF estimates the bearing- and range-rate states the policy used to have to
infer from history, which is what lets the recurrent network be replaced by an
MLP on a fully observable input.

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
_DEFAULT_DT = 0.05                            # s, integration timestep
# EKF process-noise defaults. These match the literals in the original filter
# specification; the env normally injects them from config (ekf block) so the
# single-source-of-truth / no-hardcoding rule holds.
_DEFAULT_Q_THETA = 0.01
_DEFAULT_Q_R = 1.0


class RelativeEKF:
    """
    Extended Kalman Filter for relative target tracking.
    State: x = [theta, theta_dot, r, r_dot]
    Measurements: z = [theta_noisy, r_noisy]
    """

    def __init__(
        self,
        dt: float,
        bearing_noise_std_rad: float,
        range_noise_frac: float,
        q_theta: float = _DEFAULT_Q_THETA,
        q_r: float = _DEFAULT_Q_R,
    ):
        self.dt = dt
        # Stored so update() can rebuild the range measurement variance from the
        # current range without re-hardcoding the noise fraction.
        self.range_noise_frac = float(range_noise_frac)
        self.x = np.zeros(4)          # [theta, theta_dot, r, r_dot]
        self.P = np.eye(4) * 1.0      # initial covariance

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, dt, 0,  0],
            [0,  1, 0,  0],
            [0,  0, 1, dt],
            [0,  0, 0,  1],
        ])

        # Process noise - tune these if filter is too slow or too noisy
        self.Q = np.diag([q_theta * dt, q_theta, q_r * dt, q_r])

        # Measurement matrix: we observe theta and r directly
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ])

        # Measurement noise covariance
        self.R = np.diag([
            bearing_noise_std_rad ** 2,
            (range_noise_frac * 300.0) ** 2,   # approximate at 300m; scales with range
        ])

    def reset(self, theta_init: float, r_init: float) -> None:
        self.x = np.array([theta_init, 0.0, r_init, 0.0])
        self.P = np.eye(4) * 1.0

    def update(self, theta_meas: float, r_meas: float) -> np.ndarray:
        """
        Run one EKF predict-update cycle.
        Returns estimated state [theta, theta_dot, r, r_dot].
        """
        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Update measurement noise with range-dependent range noise
        self.R[1, 1] = (self.range_noise_frac * r_meas) ** 2

        # Innovation
        z = np.array([theta_meas, r_meas])
        y = z - self.H @ x_pred

        # Kalman gain
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.x = x_pred + K @ y
        self.P = (np.eye(4) - K @ self.H) @ P_pred

        return self.x.copy()


class SensorModel:
    """Bearing + noisy range sensor with a hard field-of-view gate and an EKF."""

    def __init__(
        self,
        *,
        bearing_noise_std: float = _DEFAULT_BEARING_NOISE_STD,
        range_noise_frac: float = _DEFAULT_RANGE_NOISE_FRAC,
        fov_half_angle: float = _DEFAULT_FOV_HALF_ANGLE,
        dt: float = _DEFAULT_DT,
        ekf_q_theta: float = _DEFAULT_Q_THETA,
        ekf_q_r: float = _DEFAULT_Q_R,
    ) -> None:
        """
        Args:
            bearing_noise_std: additive bearing noise std, rad.
            range_noise_frac:  multiplicative range noise fraction (unitless).
            fov_half_angle:    field-of-view half-angle, rad.
            dt:                integration timestep, s (EKF prediction step).
            ekf_q_theta:       EKF bearing process-noise density.
            ekf_q_r:           EKF range process-noise density.
        """
        self.bearing_noise_std = float(bearing_noise_std)
        self.range_noise_frac = float(range_noise_frac)
        self.fov_half_angle = float(fov_half_angle)

        # The relative-motion estimator. observe_full() feeds it noisy
        # measurements and returns its [theta, theta_dot, r, r_dot] estimate.
        self.ekf = RelativeEKF(
            dt=float(dt),
            bearing_noise_std_rad=self.bearing_noise_std,
            range_noise_frac=self.range_noise_frac,
            q_theta=float(ekf_q_theta),
            q_r=float(ekf_q_r),
        )

        # Cached outputs of the most recent observe() call.
        # last_bearing is the NOISY observed bearing (theta_obs).
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

    def observe_full(
        self,
        interceptor_state: dict,
        intruder_state: dict,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, bool]:
        """Produce the 5-dim EKF observation and the FOV gate.

        Returns:
            obs: np.ndarray shape (5,), float32
                [0] theta_hat     - EKF estimated bearing angle (rad)
                [1] theta_dot_hat - EKF estimated bearing rate (rad/s)
                [2] r_hat         - EKF estimated range (m)
                [3] r_dot_hat     - EKF estimated range rate (m/s)
                [4] phi_self      - own heading (rad), noiseless
            in_fov: bool

        This is the current Simulink sensor slot interface. The noisy measurement
        and FOV gate come from the same logic as observe(); the EKF then turns the
        position-only measurement into a full relative-motion estimate so the
        policy input is Markov.
        """
        # 1. Get the noisy measurements (and FOV gate / caches) from observe().
        meas, in_fov = self.observe(interceptor_state, intruder_state, rng)
        theta_obs, r_obs, phi_self = float(meas[0]), float(meas[1]), float(meas[2])

        # 2. Run one EKF predict-update cycle on the measurement.
        est = self.ekf.update(theta_obs, r_obs)  # [theta, theta_dot, r, r_dot]

        # 3 + 4. Append own heading and return the 5-dim observation.
        obs = np.array(
            [est[0], est[1], est[2], est[3], phi_self], dtype=np.float32
        )
        return obs, in_fov

    def reset_filter(
        self,
        interceptor_state: dict,
        intruder_state: dict,
        rng: np.random.Generator,
    ) -> None:
        """Seed the EKF with the initial bearing and range measurements.

        Called from InterceptEnv.reset() before the first observe_full() so the
        filter starts from the spawn geometry rather than the origin, which keeps
        the first few estimates sane instead of converging from a zero prior.
        """
        meas, _ = self.observe(interceptor_state, intruder_state, rng)
        self.ekf.reset(float(meas[0]), float(meas[1]))

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
