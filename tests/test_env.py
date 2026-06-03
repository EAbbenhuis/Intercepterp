"""Sanity checks for InterceptEnv.

Covers the spec's required test surface: observation and action space shape and
bounds, the three termination/truncation paths, the EKF state estimator, the
quadratic reward with its FOV soft buffer, and a valid reset. The
terminated-vs-truncated contract from CLAUDE.md is asserted directly because it
is a hard rule (FOV loss and detonation terminate; timeout truncates).

Geometry-based tests run with sensor noise zeroed so the post-step state is
deterministic. Exact reward arithmetic is tested by driving _compute_reward with
a fully controlled internal state (self.last_obs).
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
    assert space.shape == (5,)
    assert space.dtype == np.float32
    assert np.allclose(space.low, [-np.pi, -np.pi, 0.0, -500.0, -np.pi])
    assert np.allclose(space.high, [np.pi, np.pi, 5000.0, 500.0, np.pi])


def test_observation_is_within_space(config):
    env = make_env(config)
    obs, _ = env.reset(seed=3)
    assert obs.shape == (5,)
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


# ----------------------------------------------------------------- EKF state --

def test_ekf_initialises(config):
    """After reset(), the EKF rate estimates must be finite (not nan or inf)."""
    env = make_env(config)
    obs, _ = env.reset(seed=5)
    assert obs.shape == (5,)
    assert np.isfinite(obs[1])  # theta_dot_hat
    assert np.isfinite(obs[3])  # r_dot_hat
    assert np.isfinite(obs).all()


def test_ekf_updates(config):
    """Stepping must change the EKF rate estimates (the filter is live)."""
    env = make_env(config, seed=1)
    obs, _ = env.reset(seed=1)
    prev = obs.copy()
    theta_dot_changed = False
    r_dot_changed = False
    action = np.array([0.3], dtype=np.float32)  # turn so the geometry evolves
    for _ in range(10):
        obs, _, terminated, truncated, _ = env.step(action)
        if not np.isclose(obs[1], prev[1]):
            theta_dot_changed = True
        if not np.isclose(obs[3], prev[3]):
            r_dot_changed = True
        prev = obs.copy()
        if terminated or truncated:
            break
    assert theta_dot_changed
    assert r_dot_changed


# ------------------------------------------------------ termination: events --

def test_detonation_terminates(noiseless_config):
    """r < blast_radius after a step: terminated (success), not truncated."""
    env = make_env(noiseless_config)
    # Both heading +x; interceptor 2 m behind. After one step the faster
    # interceptor closes to ~0.6 m, inside the 5 m blast radius.
    env.interceptor.x, env.interceptor.y, env.interceptor.psi = 0.0, 0.0, 0.0
    env.intruder.x, env.intruder.y, env.intruder.psi = 2.0, 0.0, 0.0

    obs, reward, terminated, truncated, info = env.step(_zero_action())

    assert info["range"] < env.blast_radius
    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "success"
    assert info["is_success"] is True
    # Terminal success dominates: bearing is ~0 and the EKF-driven step shaping
    # (a positive approach term as the range collapses) stays within a few units
    # of the +100 success reward.
    assert reward == pytest.approx(env.reward_success, abs=5.0)


def test_fov_loss_terminates(noiseless_config):
    """|theta| > FOV/2 after a step: terminated (FOV loss), not truncated."""
    env = make_env(noiseless_config)
    # Intruder at 90 deg off the interceptor boresight, well outside the 30 deg
    # FOV, and far enough that detonation cannot pre-empt it.
    env.interceptor.x, env.interceptor.y, env.interceptor.psi = 0.0, 0.0, 0.0
    env.intruder.x, env.intruder.y, env.intruder.psi = 0.0, 100.0, 0.0

    obs, reward, terminated, truncated, info = env.step(_zero_action())

    assert abs(info["bearing_true"]) > env.fov_half_angle
    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "fov_loss"
    assert info["is_success"] is False
    # Terminal FOV-loss penalty dominates the small step shaping.
    assert reward == pytest.approx(env.reward_fov_loss, abs=5.0)


def test_timeout_truncates(noiseless_config):
    """t >= timeout with target in view and out of blast range: truncated."""
    env = make_env(noiseless_config)
    env.interceptor.x, env.interceptor.y, env.interceptor.psi = 0.0, 0.0, 0.0
    env.intruder.x, env.intruder.y, env.intruder.psi = 500.0, 0.0, 0.0
    # One dt short of the limit so the in-step increment trips it exactly.
    env.t = env.episode_timeout - env.dt

    obs, reward, terminated, truncated, info = env.step(_zero_action())

    assert info["t"] >= env.episode_timeout
    assert info["range"] > env.blast_radius
    assert info["in_fov"] is True
    assert terminated is False
    assert truncated is True
    assert info["termination_reason"] == "timeout"
    # Terminal timeout penalty dominates the small step shaping.
    assert reward == pytest.approx(env.reward_timeout, abs=5.0)


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
    rew = env.config["reward"]
    assert rew["bearing_penalty"] == pytest.approx(0.003)
    assert rew["approach_reward"] == pytest.approx(0.005)
    assert rew["fov_soft_limit_deg"] == pytest.approx(20.0)
    assert rew["fov_edge_penalty"] == pytest.approx(0.05)
    assert rew["closing_reward"] == pytest.approx(0.0)
    assert env.reward_success == pytest.approx(100.0)
    assert env.reward_fov_loss == pytest.approx(-100.0)
    assert env.reward_timeout == pytest.approx(-50.0)
    # alpha caches the (now quadratic) bearing-penalty weight.
    assert env.alpha == pytest.approx(0.003)


def test_reward_values_exact(noiseless_config):
    """Drive _compute_reward with controlled self.last_obs to check arithmetic.

    last_obs = [theta_hat, theta_dot_hat, r_hat, r_dot_hat, phi_self]. All bearing
    values used here are inside the soft limit, so the FOV edge term is 0 and each
    case isolates the quadratic bearing penalty plus the approach term plus the
    terminal component.
    """
    env = make_env(noiseless_config)
    bp = env.config["reward"]["bearing_penalty"]
    ap = env.config["reward"]["approach_reward"]

    def step_reward(theta: float, r_dot: float) -> float:
        return -bp * float(theta) ** 2 + ap * max(0.0, -float(r_dot))

    # Ongoing step: bearing penalty + approach (closing), in view, out of range.
    env.last_obs = np.array([0.1, 0.0, 500.0, -20.0, 0.0], dtype=np.float32)
    env.current_range, env.in_fov, env.t = 500.0, True, 0.0
    reward, terminated, truncated = env._compute_reward()
    expected = step_reward(env.last_obs[0], env.last_obs[3])
    assert reward == pytest.approx(expected, rel=1e-5)
    assert (terminated, truncated) == (False, False)

    # Success: step reward + 100, terminated.
    env.last_obs = np.array([0.1, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    env.current_range, env.in_fov, env.t = 1.0, True, 0.0
    reward, terminated, truncated = env._compute_reward()
    expected = step_reward(env.last_obs[0], env.last_obs[3]) + 100.0
    assert reward == pytest.approx(expected, rel=1e-5)
    assert (terminated, truncated) == (True, False)
    assert env.termination_reason == "success"

    # FOV loss: step reward - 100, terminated.
    env.last_obs = np.array([0.1, 0.0, 500.0, 0.0, 0.0], dtype=np.float32)
    env.current_range, env.in_fov, env.t = 500.0, False, 0.0
    reward, terminated, truncated = env._compute_reward()
    expected = step_reward(env.last_obs[0], env.last_obs[3]) - 100.0
    assert reward == pytest.approx(expected, rel=1e-5)
    assert (terminated, truncated) == (True, False)
    assert env.termination_reason == "fov_loss"

    # Timeout: step reward - 50, truncated.
    env.last_obs = np.array([0.1, 0.0, 500.0, -5.0, 0.0], dtype=np.float32)
    env.current_range, env.in_fov = 500.0, True
    env.t = env.episode_timeout
    reward, terminated, truncated = env._compute_reward()
    expected = step_reward(env.last_obs[0], env.last_obs[3]) - 50.0
    assert reward == pytest.approx(expected, rel=1e-5)
    assert (terminated, truncated) == (False, True)
    assert env.termination_reason == "timeout"


def test_bearing_penalty_quadratic(noiseless_config):
    """The bearing penalty scales as theta^2: 0.4 rad gives ~4x the 0.2 rad value."""
    cfg = copy.deepcopy(noiseless_config)
    # Lift the soft limit so the FOV edge buffer never activates here, isolating
    # the quadratic bearing term (0.4 rad = 22.9 deg would otherwise exceed the
    # 20 deg soft limit and add an edge penalty).
    cfg["reward"]["fov_soft_limit_deg"] = 90.0
    env = InterceptEnv(cfg, ActionConfig(), curriculum_stage=1, rng_seed=0)
    env.reset(seed=0)

    def bearing_penalty(theta: float) -> float:
        # r_dot = 0 so the approach term vanishes and the reward is -w * theta^2.
        env.last_obs = np.array([theta, 0.0, 500.0, 0.0, 0.0], dtype=np.float32)
        env.current_range, env.in_fov, env.t = 500.0, True, 0.0
        reward, _, _ = env._compute_reward()
        return reward

    p02 = bearing_penalty(0.2)
    p04 = bearing_penalty(0.4)
    assert p02 < 0.0 and p04 < 0.0
    assert p04 == pytest.approx(4.0 * p02, rel=1e-3)


def test_fov_edge_penalty(noiseless_config, config):
    """A bearing between the soft limit and the hard FOV edge must add r_edge < 0."""
    env = make_env(noiseless_config)
    rew = env.config["reward"]
    soft = np.deg2rad(rew["fov_soft_limit_deg"])             # 20 deg
    hard = np.deg2rad(config["physics"]["fov_half_angle"])   # 30 deg
    theta = float(np.deg2rad(25.0))                          # between soft and hard
    assert soft < theta < hard

    # No approach term (r_dot = 0); reward = bearing penalty + edge penalty.
    env.last_obs = np.array([theta, 0.0, 500.0, 0.0, 0.0], dtype=np.float32)
    env.current_range, env.in_fov, env.t = 500.0, True, 0.0
    reward, _, _ = env._compute_reward()

    bearing_only = -rew["bearing_penalty"] * float(env.last_obs[0]) ** 2
    # The edge penalty makes the reward strictly more negative than bearing alone.
    assert reward < bearing_only
    excess = abs(float(env.last_obs[0])) - soft
    expected_edge = -rew["fov_edge_penalty"] * excess ** 2
    assert reward == pytest.approx(bearing_only + expected_edge, rel=1e-4)


def test_reward_priority_success_over_fov_loss(noiseless_config):
    """If detonation and FOV loss coincide, success must win (spec ordering)."""
    env = make_env(noiseless_config)
    env.last_obs = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    env.current_range, env.in_fov, env.t = 1.0, False, 0.0
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
    assert obs.shape == (5,)
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


# ----------------------------------------------- zero-action baseline floor --

def test_zero_action_baseline(config):
    """Zero-action policy must intercept < 20% at bearing_offset_max = 25 deg.

    Sanity check that the spawn geometry is not trivially solvable by flying
    straight: a learned bearing-only policy must clear this floor. Independent of
    the reward (only the kinematics and termination geometry matter).
    """
    cfg = copy.deepcopy(config)
    cfg["init"]["bearing_offset_max"] = 25.0
    env = InterceptEnv(cfg, ActionConfig(), curriculum_stage=1, rng_seed=0)

    successes = 0
    n_eps = 50
    info: dict = {}
    for ep in range(n_eps):
        env.reset(seed=ep)
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(_zero_action())
            done = terminated or truncated
        if info.get("termination_reason") == "success":
            successes += 1

    success_rate = successes / n_eps
    assert success_rate < 0.20


# ------------------------------------------------- tuning mode + performance --

def test_tuning_range_override(noiseless_config):
    """A 300 m tuning spawn must place the target at r_hat in 300 +- 10%.

    Sensor noise is removed so the EKF estimate equals the true spawn range
    exactly; the spawn band is U(range_mean - range_std, range_mean + range_std)
    = U(270, 330), which is what the tuning override (range_mean 300, range_std
    30) produces. r_hat is observation index 2 in the 5-dim EKF observation.
    """
    cfg = copy.deepcopy(noiseless_config)
    cfg["init"]["range_mean"] = 300.0
    cfg["init"]["range_std"] = 30.0
    env = InterceptEnv(cfg, ActionConfig(), curriculum_stage=1, rng_seed=0)

    for seed in range(25):
        obs, _ = env.reset(seed=seed)
        r_hat = float(obs[2])
        assert 270.0 <= r_hat <= 330.0


def test_eval_stage_flag(config):
    """eval.py must route its --stage flag straight to InterceptEnv.

    Guards the regression where eval defaulted to the final stage: the CLI
    default is 1, and build_eval_env (the single place the stage reaches the env)
    must construct an InterceptEnv whose curriculum_stage equals the requested
    stage for both an easy (1) and the hardest (3) stage.
    """
    from eval.eval import build_eval_env, parse_args

    # CLI default stage is 1 (not the final stage); --stage overrides it.
    assert parse_args(["--model", "x.zip"]).stage == 1
    assert parse_args(["--model", "x.zip", "--stage", "3"]).stage == 3

    for stage in (1, 3):
        env = build_eval_env(config, stage=stage, seed=0)
        assert env.curriculum_stage == stage
        env.close()


def test_no_render_in_step(config):
    """With render_mode=None, stepping must never create a matplotlib figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = make_env(config)  # make_env leaves render_mode at its None default
    assert env.render_mode is None

    before = len(plt.get_fignums())
    for _ in range(100):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
    after = len(plt.get_fignums())

    assert after <= before
