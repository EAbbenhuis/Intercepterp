"""Sanity checks for InterceptEnv.

Covers the spec's required test surface: observation and action space shape and
bounds, the three termination/truncation paths, exact reward values, and a valid
reset. The terminated-vs-truncated contract from CLAUDE.md is asserted directly
because it is a hard rule (FOV loss and detonation terminate; timeout truncates).

Geometry-based tests run with sensor noise zeroed so the post-step state is
deterministic. Exact reward arithmetic is tested by driving _compute_reward with
a fully controlled internal state.
"""
from __future__ import annotations

import copy
import pathlib

import gymnasium
import numpy as np
import pytest
import yaml

from envs.intercept_env import ActionConfig, InterceptEnv

_CONFIG_PATH = pathlib.Path(__file__).resolve().parents[1] / "config" / "defaults.yaml"


# --------------------------------------------------------------------- setup --

@pytest.fixture
def config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def noiseless_config(config: dict) -> dict:
    """Config with sensor noise removed, for deterministic geometry tests."""
    cfg = copy.deepcopy(config)
    cfg["sensor"]["bearing_noise_std"] = 0.0
    cfg["sensor"]["range_noise_frac"] = 0.0
    return cfg


def make_env(cfg: dict, stage: int = 1, seed: int = 0) -> InterceptEnv:
    env = InterceptEnv(cfg, ActionConfig(), curriculum_stage=stage, rng_seed=seed)
    env.reset(seed=seed)
    return env


def _zero_action() -> np.ndarray:
    return np.zeros(1, dtype=np.float32)


# ------------------------------------------------------------ spaces: shape --

def test_observation_space_shape_and_bounds(config):
    env = make_env(config)
    space = env.observation_space
    assert isinstance(space, gymnasium.spaces.Box)
    assert space.shape == (3,)
    assert space.dtype == np.float32
    assert np.allclose(space.low, [-np.pi, 0.0, -np.pi])
    assert np.allclose(space.high, [np.pi, 5000.0, np.pi])


def test_observation_is_within_space(config):
    env = make_env(config)
    obs, _ = env.reset(seed=3)
    assert obs.shape == (3,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)


def test_observation_stays_in_bounds_over_rollout(config):
    """Noise plus clipping must keep every observation inside the Box."""
    env = make_env(config, seed=7)
    obs, _ = env.reset(seed=7)
    assert env.observation_space.contains(obs)
    for _ in range(200):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            obs, _ = env.reset()
            assert env.observation_space.contains(obs)


def test_action_space_shape_and_bounds(config):
    env = make_env(config)
    space = env.action_space
    assert isinstance(space, gymnasium.spaces.Box)
    assert space.shape == (ActionConfig().dim,)
    assert space.shape == (1,)
    assert space.dtype == np.float32
    assert np.allclose(space.low, -1.0)
    assert np.allclose(space.high, 1.0)


def test_action_config_dim():
    assert ActionConfig().dim == 1
    assert ActionConfig(turn_rate=True).dim == 1


# ------------------------------------------------------ termination: events --

def test_detonation_terminates(noiseless_config):
    """r < blast_radius after a step: terminated (success), not truncated."""
    env = make_env(noiseless_config)
    # Both heading +x; interceptor 2 m behind. After one step the faster
    # interceptor closes to ~0.6 m, inside the 5 m blast radius.
    env.interceptor.x, env.interceptor.y, env.interceptor.psi = 0.0, 0.0, 0.0
    env.intruder.x, env.intruder.y, env.intruder.psi = 2.0, 0.0, 0.0
    # Neutralise the closing reward: with the previous observed range driven to
    # 0, this step's range is larger, so closing = max(0, 0 - r) = 0 and the
    # reward isolates the terminal component.
    env.current_r_obs = 0.0

    obs, reward, terminated, truncated, info = env.step(_zero_action())

    assert info["range"] < env.blast_radius
    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "success"
    assert info["is_success"] is True
    # Bearing is exactly 0 in this collinear geometry and closing is zeroed,
    # so reward is purely +100.
    assert reward == pytest.approx(env.reward_success)


def test_fov_loss_terminates(noiseless_config):
    """|theta| > FOV/2 after a step: terminated (FOV loss), not truncated."""
    env = make_env(noiseless_config)
    # Intruder at 90 deg off the interceptor boresight, well outside the 30 deg
    # FOV, and far enough that detonation cannot pre-empt it.
    env.interceptor.x, env.interceptor.y, env.interceptor.psi = 0.0, 0.0, 0.0
    env.intruder.x, env.intruder.y, env.intruder.psi = 0.0, 100.0, 0.0
    # Neutralise the closing reward so the assertion isolates the bearing penalty
    # plus the terminal FOV-loss component.
    env.current_r_obs = 0.0

    obs, reward, terminated, truncated, info = env.step(_zero_action())

    assert abs(info["bearing_true"]) > env.fov_half_angle
    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "fov_loss"
    assert info["is_success"] is False
    # reward = r_step + fov_loss, with r_step = -alpha * |theta_obs| (closing 0).
    expected = -env.alpha * abs(info["bearing_obs"]) + env.reward_fov_loss
    assert reward == pytest.approx(expected)


def test_timeout_truncates(noiseless_config):
    """t >= timeout with target in view and out of blast range: truncated."""
    env = make_env(noiseless_config)
    env.interceptor.x, env.interceptor.y, env.interceptor.psi = 0.0, 0.0, 0.0
    env.intruder.x, env.intruder.y, env.intruder.psi = 500.0, 0.0, 0.0
    # One dt short of the limit so the in-step increment trips it exactly.
    env.t = env.episode_timeout - env.dt
    # Neutralise the closing reward so the assertion isolates the timeout penalty.
    env.current_r_obs = 0.0

    obs, reward, terminated, truncated, info = env.step(_zero_action())

    assert info["t"] >= env.episode_timeout
    assert info["range"] > env.blast_radius
    assert info["in_fov"] is True
    assert terminated is False
    assert truncated is True
    assert info["termination_reason"] == "timeout"
    # Bearing is 0 in this collinear geometry and closing is zeroed, so reward
    # is purely the timeout penalty.
    assert reward == pytest.approx(env.reward_timeout)


def test_terminated_truncated_are_never_both_true(config):
    env = make_env(config, seed=11)
    env.reset(seed=11)
    for _ in range(1200):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert not (terminated and truncated)
        if terminated or truncated:
            break


# ---------------------------------------------------------- reward: exact --

def test_reward_constants_match_spec(config):
    env = make_env(config)
    assert env.alpha == pytest.approx(0.05)
    assert env.reward_success == pytest.approx(100.0)
    assert env.reward_fov_loss == pytest.approx(-100.0)
    assert env.reward_timeout == pytest.approx(-100.0)
    assert env.config["reward"]["closing_reward"] == pytest.approx(0.015)


def test_reward_values_exact(noiseless_config):
    """Drive _compute_reward with controlled state to check exact arithmetic.

    reset() leaves prev_r_obs == current_r_obs, so the closing term is 0 here and
    each case isolates the bearing penalty plus the terminal component.
    """
    env = make_env(noiseless_config)

    # Ongoing step: only the bearing penalty applies.
    env.range, env.in_fov, env.t, env.sensor.last_bearing = 500.0, True, 0.0, 0.3
    reward, terminated, truncated = env._compute_reward()
    assert reward == pytest.approx(-0.05 * 0.3)
    assert (terminated, truncated) == (False, False)

    # Success: penalty + 100, terminated.
    env.range, env.in_fov, env.t, env.sensor.last_bearing = 1.0, True, 0.0, 0.2
    reward, terminated, truncated = env._compute_reward()
    assert reward == pytest.approx(-0.05 * 0.2 + 100.0)
    assert (terminated, truncated) == (True, False)
    assert env.termination_reason == "success"

    # FOV loss: penalty - 100, terminated.
    env.range, env.in_fov, env.t, env.sensor.last_bearing = 500.0, False, 0.0, 0.6
    reward, terminated, truncated = env._compute_reward()
    assert reward == pytest.approx(-0.05 * 0.6 - 100.0)
    assert (terminated, truncated) == (True, False)
    assert env.termination_reason == "fov_loss"

    # Timeout: penalty - 100, truncated.
    env.range, env.in_fov, env.sensor.last_bearing = 500.0, True, 0.1
    env.t = env.episode_timeout
    reward, terminated, truncated = env._compute_reward()
    assert reward == pytest.approx(-0.05 * 0.1 - 100.0)
    assert (terminated, truncated) == (False, True)
    assert env.termination_reason == "timeout"


def test_closing_reward_positive_when_range_decreases(noiseless_config):
    """A drop in observed range between consecutive steps must pay a bonus."""
    env = make_env(noiseless_config)
    # Ongoing, in-view, collinear (zero bearing) so only the closing term is left.
    env.range, env.in_fov, env.t, env.sensor.last_bearing = 500.0, True, 0.0, 0.0
    env.prev_r_obs, env.current_r_obs = 600.0, 500.0  # 100 m of closing

    reward, terminated, truncated = env._compute_reward()

    closing_coeff = env.config["reward"]["closing_reward"]
    assert reward > 0.0
    assert reward == pytest.approx(closing_coeff * 100.0)
    assert (terminated, truncated) == (False, False)


def test_closing_reward_zero_when_range_increases(noiseless_config):
    """A rise in observed range must not be rewarded (and never penalised)."""
    env = make_env(noiseless_config)
    env.range, env.in_fov, env.t, env.sensor.last_bearing = 500.0, True, 0.0, 0.0
    env.prev_r_obs, env.current_r_obs = 500.0, 600.0  # range grew by 100 m

    reward, terminated, truncated = env._compute_reward()

    # Bearing is 0 and the closing term clips to 0, so the step reward is exactly 0.
    assert reward == pytest.approx(0.0)
    assert (terminated, truncated) == (False, False)


def test_reward_priority_success_over_fov_loss(noiseless_config):
    """If detonation and FOV loss coincide, success must win (spec ordering)."""
    env = make_env(noiseless_config)
    env.range, env.in_fov, env.t, env.sensor.last_bearing = 1.0, False, 0.0, 0.0
    reward, terminated, truncated = env._compute_reward()
    assert reward == pytest.approx(100.0)
    assert env.termination_reason == "success"
    assert (terminated, truncated) == (True, False)


def test_action_scaling_to_psi_dot_max(noiseless_config):
    """action = +1 / -1 must map to +/- psi_dot_max turn over one dt."""
    env = make_env(noiseless_config)
    psi_dot_max = env.interceptor.psi_dot_max

    env.interceptor.psi = 0.0
    env.interceptor.apply_action(np.array([1.0], dtype=np.float32))
    assert env.interceptor.psi == pytest.approx(psi_dot_max * env.dt)

    env.interceptor.psi = 0.0
    env.interceptor.apply_action(np.array([-1.0], dtype=np.float32))
    assert env.interceptor.psi == pytest.approx(-psi_dot_max * env.dt)


# ------------------------------------------------------------- reset: valid --

def test_reset_returns_valid_obs_and_info(config):
    env = InterceptEnv(config, ActionConfig(), curriculum_stage=1, rng_seed=0)
    obs, info = env.reset(seed=0)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (3,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)

    assert isinstance(info, dict)
    for key in ("t", "range", "in_fov", "stage", "termination_reason", "is_success"):
        assert key in info
    assert info["t"] == 0.0
    assert info["stage"] == 1
    # Target starts inside the FOV (interceptor is aimed at the intruder).
    assert info["in_fov"] is True
    assert info["range"] > 0.0


def test_reset_is_seed_reproducible(config):
    env_a = InterceptEnv(config, ActionConfig(), rng_seed=0)
    env_b = InterceptEnv(config, ActionConfig(), rng_seed=0)
    obs_a, _ = env_a.reset(seed=123)
    obs_b, _ = env_b.reset(seed=123)
    assert np.array_equal(obs_a, obs_b)

    # Same seed, same action sequence, identical trajectory.
    for _ in range(20):
        a = env_a.action_space.sample()
        sa = env_a.step(a)
        sb = env_b.step(a)
        assert np.array_equal(sa[0], sb[0])
        assert sa[1] == sb[1]
