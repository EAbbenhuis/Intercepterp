"""Sanity checks for InterceptEnv.

Covers the spec's required test surface: observation and action space shape and
bounds, the three termination/truncation paths, the EKF state estimator, the
v5 bearing-rate reward (rate penalty + bearing penalty), and a valid reset. The
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
    """The committed reward constants must match the v5 reward block.

    These expected values track config/defaults.yaml and must be updated again if
    the reward block changes.
    """
    env = make_env(config)
    rew = env.config["reward"]
    assert rew["bearing_rate_penalty"] == pytest.approx(0.667)
    assert rew["bearing_penalty"] == pytest.approx(0.022)
    assert env.reward_success == pytest.approx(100.0)
    assert env.reward_fov_loss == pytest.approx(-100.0)
    assert env.reward_timeout == pytest.approx(-50.0)
    # The v4 shaping keys were removed and must no longer be present.
    for removed in (
        "approach_reward",
        "fov_soft_limit_deg",
        "fov_edge_penalty",
        "closing_reward",
    ):
        assert removed not in rew
    # alpha caches the primary bearing-rate weight, beta the secondary bearing weight.
    assert env.alpha == pytest.approx(0.667)
    assert env.beta == pytest.approx(0.022)


def test_bearing_rate_reward(noiseless_config):
    """The v5 step reward is two non-positive terms: a bearing-rate penalty
    -alpha*|theta_dot| (primary) and a bearing penalty -beta*|theta| (secondary).

    last_obs = [theta_hat, theta_dot_hat, r_hat, r_dot_hat, phi_self]. A controlled
    non-terminal state isolates the step reward arithmetic; a real step then
    confirms the same decomposition holds on the env's own observation.
    """
    env = make_env(noiseless_config, seed=0)
    rew = env.config["reward"]
    alpha = rew["bearing_rate_penalty"]
    beta = rew["bearing_penalty"]

    # Controlled non-terminal state with a clear off-boresight bearing and a clear
    # bearing rate, so each shaping term is strictly negative.
    theta, theta_dot = 0.20, 0.10
    env.last_obs = np.array([theta, theta_dot, 500.0, 0.0, 0.0], dtype=np.float32)
    env.current_range, env.in_fov, env.t = 500.0, True, 0.0
    reward, terminated, truncated = env._compute_reward()
    assert (terminated, truncated) == (False, False)

    rate_term = -alpha * abs(theta_dot)
    bearing_term = -beta * abs(theta)
    assert rate_term < 0.0
    assert bearing_term < 0.0
    # The bearing-rate term is the primary signal and dominates the bearing term.
    assert abs(rate_term) > abs(bearing_term)
    # The step reward is exactly the sum of the two terms (no terminal component).
    assert reward == pytest.approx(rate_term + bearing_term, rel=1e-5)

    # A real step returns the same decomposition built from its own last_obs.
    obs, step_reward, term, trunc, _ = env.step(np.array([0.5], dtype=np.float32))
    if not (term or trunc):
        expected = -alpha * abs(float(obs[1])) - beta * abs(float(obs[0]))
        assert step_reward == pytest.approx(expected, rel=1e-5, abs=1e-9)


def test_collision_course_better_than_pursuit(noiseless_config):
    """A collision course must out-reward a tail chase under the v5 reward.

    Head-on geometry holds the target on a constant near-zero bearing (zero
    bearing rate), the signature of a lead-pursuit collision course, so the
    bearing-rate penalty stays ~0. A tail chase sits off boresight with a drifting
    bearing, so it pays the bearing-rate penalty every step. Twenty zero-action
    steps from each geometry must leave the collision course with the strictly
    higher cumulative reward.
    """
    def rollout(setup) -> float:
        env = make_env(noiseless_config, seed=0)
        setup(env)
        # Re-seed the EKF from the overridden spawn geometry so the first estimate
        # reflects the test geometry, not the random reset spawn.
        env.sensor.reset_filter(
            env.interceptor.get_state(), env.intruder.get_state(), env.np_random
        )
        total = 0.0
        for _ in range(20):
            _, reward, terminated, truncated, _ = env.step(_zero_action())
            total += reward
            if terminated or truncated:
                break
        return total

    def head_on(env: InterceptEnv) -> None:
        # Interceptor heading +x; intruder 400 m ahead heading -x (straight at it).
        # Both stay on y = 0, so the bearing holds at ~0 with ~0 bearing rate.
        env.interceptor.x, env.interceptor.y, env.interceptor.psi = 0.0, 0.0, 0.0
        env.intruder.x, env.intruder.y, env.intruder.psi = 400.0, 0.0, np.pi

    def tail_chase(env: InterceptEnv) -> None:
        # Interceptor heading +x; intruder 400 m out at ~0.3 rad off boresight,
        # flying +x (lagging pursuit). The bearing drifts, so theta_dot != 0.
        d, off = 400.0, 0.3
        env.interceptor.x, env.interceptor.y, env.interceptor.psi = 0.0, 0.0, 0.0
        env.intruder.x = float(d * np.cos(off))
        env.intruder.y = float(d * np.sin(off))
        env.intruder.psi = 0.0

    r_head_on = rollout(head_on)
    r_tail_chase = rollout(tail_chase)

    assert r_head_on > r_tail_chase


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
