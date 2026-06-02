"""Abstract base environment for Intercepterp.

BaseEnv pins down the contract that every environment variant must satisfy:
each concrete env supplies an observation builder, an info builder, and a
reward/termination computation. The Gymnasium reset/step/render methods are
implemented by the concrete subclass (see InterceptEnv).

This module also hosts wrap_to_pi, the single angle-wrapping helper used by the
sensor and both agents, so that there is one canonical definition rather than
several copies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import gymnasium
import numpy as np

# Two pi, cached. Used by wrap_to_pi on every sensor read and agent step.
_TWO_PI = 2.0 * np.pi


def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle (radians) into the half-open interval [-pi, pi).

    Works for scalars and numpy arrays. Exact multiples of pi map to -pi, which
    keeps the result inside the closed interval [-pi, pi] used for the bearing
    and heading observation bounds.
    """
    return (angle + np.pi) % _TWO_PI - np.pi


class BaseEnv(gymnasium.Env, ABC):
    """Abstract Gymnasium environment.

    Subclasses must define observation_space and action_space, implement the
    standard reset/step/render API, and provide the three hooks below. Keeping
    these abstract guarantees that any future env variant (for example a 3D or
    multi-target version) exposes the same internal structure that the training,
    eval and viz code relies on.
    """

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        """Return the current observation as a float32 array."""
        raise NotImplementedError

    @abstractmethod
    def _get_info(self) -> dict:
        """Return the auxiliary info dict for the current step."""
        raise NotImplementedError

    @abstractmethod
    def _compute_reward(self) -> tuple[float, bool, bool]:
        """Return (reward, terminated, truncated) for the current state."""
        raise NotImplementedError
