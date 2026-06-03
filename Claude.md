# Intercepterp — CLAUDE.md

Counter-UAS bearing-only RL interception. Fixed-wing interceptor learns to
destroy a Shahed-136 class intruder using bearing angle and noisy range only.
Named after the Terp. Obviously.

---

## Non-negotiable rules

- **Never create a virtual environment.** Use the system Python directly.
  No `venv`, no `conda`, no `uv venv`, no `.venv`. Ever.
- No em dashes anywhere: not in code, comments, docstrings, or markdown.
- All internal units are SI: metres, seconds, radians.
  Degrees only in config YAML and printed terminal output.
- `rng: np.random.Generator` is passed explicitly to every class that needs
  randomness. No global random state. Seed flows from `InterceptEnv.reset(seed=)`.
- `terminated` vs `truncated` must be correct. FOV loss and detonation are
  `terminated=True`. Timeout is `truncated=True`. Never mix these up.
- Action space bounds are always `[-1, 1]`. The env scales to physical units
  internally. Never change this contract.
- The three Simulink interface points (plant slot, sensor slot, policy export)
  must never be broken by any refactor. See spec section 9.
- `main` branch is always runnable. Work on feature branches.
  Branch names: `feature/description`. Conventional commits.
- After every merge into main, push to origin. 
  main is the GitHub default branch. Never change the default branch.

---

## File map

```
intercepterp/
├── CLAUDE.md                   # this file
├── README.md                   # 3-sentence project summary
├── config/
│   └── defaults.yaml           # all parameters, single source of truth
├── envs/
│   ├── base_env.py             # abstract BaseEnv(gymnasium.Env)
│   ├── intercept_env.py        # InterceptEnv — main environment
│   ├── agents/
│   │   ├── intruder.py         # Intruder: Dubins + curriculum evasion
│   │   └── interceptor.py      # Interceptor: Dubins + apply_action()
│   └── sensor.py               # SensorModel: bearing + range + FOV gate + EKF
├── training/
│   ├── train.py                # PPO (MLP) entry point
│   ├── curriculum.py           # CurriculumScheduler
│   └── callbacks.py            # EvalCallback, CurriculumCallback
├── eval/
│   ├── eval.py                 # run N episodes, write eval_results.json
│   └── metrics.py              # intercept rate, time-to-intercept, etc.
├── viz/
│   ├── dashboard.py            # live training dashboard (matplotlib)
│   └── replay.py               # replay saved episode, plot trajectory
├── tests/
│   └── test_env.py             # sanity checks: obs space, reward, termination
└── runs/                       # gitignored — timestamped run outputs
```

---

## Key interfaces (never break these)

### Observation vector — shape (5,) float32 (EKF estimate, since 2026-06-03 v4)
| Index | Variable      | Units | Source                              |
|-------|---------------|-------|-------------------------------------|
| 0     | theta_hat     | rad   | EKF estimate of bearing             |
| 1     | theta_dot_hat | rad/s | EKF estimate of bearing rate        |
| 2     | r_hat         | m     | EKF estimate of range               |
| 3     | r_dot_hat     | m/s   | EKF estimate of range rate          |
| 4     | phi_self      | rad   | own heading, noiseless              |

The EKF (RelativeEKF in sensor.py) consumes the noisy bearing N(0, 0.5 deg) and
range N(0, 0.05 * r) measurements. The legacy 3-dim observe() (theta_obs, r_obs,
phi_self) is retained for backward compatibility.

### Action vector — shape (1,) float32, bounds [-1, 1]
| Index | Variable   | Physical meaning            |
|-------|------------|-----------------------------|
| 0     | turn_rate  | scaled to psi_dot_max deg/s |

To add thrust later: append index 1 in ActionConfig and one line in
Interceptor.apply_action(). Nothing else changes.

### SensorModel.observe_full() signature (current Simulink sensor slot)
```python
def observe_full(
    self,
    interceptor_state: dict,
    intruder_state: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    # returns (obs_5dim, in_fov)  -- EKF estimate + own heading
```
The 3-dim observe() is kept for backward compatibility but the env now uses
observe_full(). reset_filter() seeds the EKF in InterceptEnv.reset().

### Reward (v4, since 2026-06-03: quadratic + EKF, no closing term)
```
r_step     = -0.003 * theta_hat^2                          every step
           + 0.005 * max(0, -r_dot_hat)                    approach reward
           - 0.05  * (|theta_hat| - 20deg)^2 if |theta_hat| > 20deg   FOV soft buffer
r_terminal = +100  if r < 5m                               (terminated, success)
           = -100  if |theta| > 30 deg                     (terminated, FOV loss)
           = -50   if t > 30s                              (truncated, timeout)
```
theta_hat, r_dot_hat are EKF outputs; r (termination) is the true range.

---

## Curriculum stages

| Stage | Intruder behaviour         | Advance when                              |
|-------|----------------------------|-------------------------------------------|
| 1     | Straight flight            | intercept rate > 70% over 100 eps         |
|       |                            | OR no improvement for 500 eps             |
| 2     | Sinusoidal heading (~8s)   | intercept rate > 65% over 100 eps         |
|       |                            | OR no improvement for 500 eps             |
| 3     | Reactive evasion           | final stage                               |

---

## Run output structure

Every training run saves to `runs/YYYYMMDD_HHMMSS/`:
```
runs/20260602_143200/
├── config.yaml          # snapshot of config used
├── best_model.zip       # saved by EvalCallback
├── final_model.zip      # saved at end of training
├── eval_results.json    # written by eval/eval.py
└── tensorboard/         # SB3 tensorboard logs
```

`runs/` is in `.gitignore`. Models are never committed.

---

## Session log

Append a new entry after every Claude Code session.

```
## YYYY-MM-DD
- What was built or changed
- Any decisions made that deviate from spec (justify them)
- Known issues or next steps
```

## 2026-06-02
- Built the full repository from scratch per INTERCEPTERP_SPEC.md: config/defaults.yaml,
  the env stack (base_env, agents/intruder, agents/interceptor, sensor, intercept_env),
  a 15-case pytest suite, training (curriculum, callbacks, train), evaluation
  (metrics, eval), visualisation (replay, dashboard), README, and .gitignore.
- Validation: env passes SB3 check_env with no warnings; all 15 tests pass; a short
  RecurrentPPO smoke run trained, evaluated (wrote eval_results.json), and replayed
  (saved a GIF) end to end. Full 5e6-step training was not run here.
- Decisions and spec ambiguities (all flagged in code comments):
  - Action scaling: env keeps action_space at [-1, 1] (hard rule) and scales to
    physical rad/s inside Interceptor.apply_action (clip to [-1, 1], then multiply by
    psi_dot_max). This reconciles the [-1, 1] contract with the spec 4.6 pseudocode,
    which folded clip-and-scale into one clip-to-psi_dot_max line.
  - FOV gate uses the TRUE bearing, not the noisy one: a camera only produces a
    detection when the target is physically in frame, so noise cannot trigger spurious
    FOV-loss termination. The step penalty still uses the NOISY theta_obs (spec 5).
  - Spawn range: spec 4.4 says U(900, 1100) while section 7 names range_mean and
    range_std; implemented U(mean - std, mean + std), which equals U(900, 1100) for
    the defaults and follows 4.4's explicit uniform statement.
  - Interceptor spawn heading points exactly at the intruder ("roughly toward"
    interpreted as exactly toward), guaranteeing the target is in FOV at t = 0.
  - Stage-2 weave amplitude was unspecified; tied it to A = psi_dot_max * T / (2*pi)
    so the peak turn rate equals the intruder psi_dot_max and respects the bound.
  - Added intruder.psi_dot_max (15 deg/s) and intruder.sin_period (8 s) to config:
    stated in spec 4.3 but missing from the section 7 YAML; placed in config to avoid
    hardcoding (no-hardcoding rule).
  - Curriculum stage is pushed to the live envs via env_method("set_wrapper_attr", ...)
    so it reaches InterceptEnv through the Monitor wrapper (plain VecEnv.set_attr sets
    the attribute on the wrapper, where reset never reads it).
  - Infra: ActionConfig lives in envs/intercept_env.py (no separate file in the map);
    added conftest.py and package __init__.py files plus a sys.path bootstrap in entry
    scripts so both pytest and `python eval/eval.py` style invocation import cleanly.
- Git: initial implementation committed on main; this session log committed on
  feature/initial-build (the spec's git step asked for a commit on that branch).
- Next steps: run full 5e6-step training and tune curriculum thresholds (0.70/0.65);
  add ONNX export for the Simulink policy slot (spec section 9); consider adding
  bearing-rate (index 3) only if the LSTM cannot infer it from history.

## 2026-06-02
- Added output.base_dir to config and made training output paths Kaggle-aware.
- Added device selection prints and GPU name logging in training entry point.
- Added requirements.txt for Kaggle installation.
- Tests: python -m pytest tests/test_env.py

## 2026-06-02 (reward v2: loitering fix)
- Behavioural problem: the policy converged to a loitering local minimum, holding
  the target near 10 deg bearing and absorbing the soft timeout penalty instead of
  closing. Applied targeted reward and training fixes (no kinematics, observation,
  sensor, curriculum, or file-structure changes).
- config/defaults.yaml reward block: bearing_penalty 0.1 -> 0.05 (the loitering
  incentive scaled with this penalty, so orbiting at a small bearing was cheap;
  halving it weakens that pull). Added closing_reward 0.015 per metre of observed
  range decrease per step (gives a dense gradient toward closing, which a sparse
  +100-only success signal failed to provide). timeout -50 -> -100 (the soft
  penalty made fixed-range loitering a stable equilibrium; hardening it removes
  that equilibrium so timing out is strictly worse than attempting an intercept).
- envs/intercept_env.py: _compute_reward now adds closing = max(0, prev_r_obs -
  current_r_obs) * closing_reward, using the NOISY observed range (consistent with
  the GPS-denied sensor model) and never penalising range increase, so evasive
  geometry cannot produce a negative closing term. Added prev_r_obs/current_r_obs
  instance vars: prev_r_obs seeds in reset() to the initial observed range, and
  step() rolls the window forward (prev <- current, current <- obs[1]) before
  rewarding. The termination_reason assignments were retained inside
  _compute_reward because eval, the info dict, and the curriculum is_success flag
  all read them; dropping them (as the literal patch snippet did) would silently
  break curriculum advancement, which the hard rules forbid.
- training/train.py: replaced the implicit fixed entropy coefficient with
  ent_coef=linear_schedule(0.02 -> 0.001 over the first 60% of training). High
  early entropy keeps exploration alive long enough to discover closing behaviour
  before the policy commits; the decay then lets it sharpen the intercept.
- tests/test_env.py: timeout penalty assertions -50 -> -100; bearing-penalty
  multipliers 0.1 -> 0.05; added a closing_reward constant check; added two tests
  (closing reward positive when range decreases, zero when range increases). The
  three geometry-override termination tests set current_r_obs = 0 before stepping
  so the spurious range jump from manual respawn cannot leak into the closing term
  and the assertions still isolate the terminal reward. All 17 tests pass.

## 2026-06-02 (performance and fast tuning mode)
- Throughput-only changes plus a diagnostic tuning mode. No reward logic, env
  kinematics, sensor model, curriculum thresholds, or observation space touched.
- config/defaults.yaml training block: n_envs 8 -> 32, n_steps 2048 -> 512,
  n_epochs 10 -> 4 (more parallel rollout workers with shorter, cheaper PPO
  updates raises wall-clock throughput; batch_size, lr, clip_range, and the LSTM
  sizes are unchanged). total_timesteps stays at the 5e6 production value.
- Added a tuning block: enabled false (must stay false when committed),
  total_timesteps 300k, range_mean 300 / range_std 30, freeze_curriculum true,
  stage 1. It is the single source of truth for the diagnostic run.
- training/train.py: added a --tuning store_true flag. When set, train() prints a
  visible banner and overrides config init.range_mean, init.range_std, and
  training.total_timesteps from the tuning block, pins the env curriculum_stage to
  tuning.stage, and freezes the curriculum by building a single-stage scheduler
  (thresholds=[None]) that is final from the start and never advances. The
  config thresholds are not mutated; the freeze is in-memory and tuning-only.
  Without --tuning, behaviour is byte-for-byte identical to before.
- envs/intercept_env.py: performance audit of step() and _compute_reward(). All
  three checks passed with no code change required (matplotlib is confined to the
  render_mode-guarded render(); step() does no disk I/O; observe() and the Dubins
  updates use scalar numpy or plain math with no per-step list-to-array of data
  and no Python loops over arrays). The audit result is documented as a comment at
  the top of step().
- tests/test_env.py: added test_tuning_range_override (noiseless 300 m spawn keeps
  r_obs in [270, 330] across 25 seeds) and test_no_render_in_step (100 steps with
  render_mode=None create no matplotlib figures, checked via get_fignums). All 19
  tests pass.
- WARNING: tuning mode must never be committed with enabled: true. It is a local
  diagnostic switch only; a production run requires enabled false (the default)
  and no --tuning flag.

## 2026-06-02 (entropy schedule fix)
- Bug: passing the callable linear_schedule to RecurrentPPO's ent_coef raised
  TypeError: unsupported operand type(s) for *: 'function' and 'Tensor' at the
  first update. sb3_contrib RecurrentPPO (like base SB3) only schedules
  learning_rate and clip_range; ent_coef must be a plain float, used directly as
  self.ent_coef * entropy_loss in train(), so a function there fails.
- Fix (training/train.py): removed the linear_schedule function entirely and pass
  a fixed ent_coef=0.01 to RecurrentPPO. Entropy decay is now applied at runtime.
- training/callbacks.py: added EntropyDecayCallback(BaseCallback). Each _on_step
  it sets self.model.ent_coef to a float linearly annealed from initial_value
  0.02 to final_value 0.001 over the first end_fraction (0.6) of training, then
  holds at final_value. Exported it in __all__.
- training/train.py wires it in: instantiate EntropyDecayCallback with
  total_timesteps equal to the resolved run length (tuning or production) so the
  decay is proportional in both modes, and pass callback=[callbacks, entropy_cb]
  to model.learn (SB3 wraps the list in a CallbackList). Net behaviour matches the
  intended 0.02 -> 0.001 schedule, but now driven by a callback that mutates a
  float instead of an unsupported callable.
- No reward, env, sensor, or observation-space changes. All 19 tests pass; the
  decay was unit-checked at 0%, 30%, 60%, and 100% progress (0.02, 0.0105, 0.001,
  0.001).

## 2026-06-02 (eval stage default + reward test reconcile)
- Bug: eval/eval.py defaulted the evaluation stage to the final curriculum stage
  via stage = args.stage or len(config["curriculum"]["thresholds"]) (== 3), so a
  plain eval ran the full-difficulty task even for a stage-1 policy. Fixed to
  default to stage 1: --stage now has default=1 and stage = args.stage is passed
  straight to InterceptEnv. --stage still overrides for stages 2 and 3.
- Sweep integration: train.py does not call eval.py as a subprocess; in-training
  evaluation is callback-driven (SB3 EvalCallback on eval_env), and eval_env is
  already built with the correct stage (tuning freezes at stage 1, otherwise the
  CLI/curriculum stage kept in sync by CurriculumCallback), so no callback change
  was needed. The only eval.py touch point was the end-of-run hint, which now
  prints --stage {scheduler.current_stage}: that is 1 in tuning mode (frozen
  scheduler) and the current curriculum stage otherwise, matching the spec.
- Reward tests: test_reward_constants_match_spec and test_reward_values_exact
  still encoded the reward-v2 constants (bearing_penalty 0.05, timeout -100),
  which no longer match the committed reward config (bearing_penalty 0.1,
  closing_reward 0.15, timeout -50). Per an explicit decision this session, the
  two tests were reconciled to the current config so the suite is green. These
  expected constants track config and must be updated again if the reward block
  changes. All 19 tests pass.

## 2026-06-02 (VecNormalize reward scaling + eval --stage VecNormalize load)
- Two issues from the 300k training logs: (1) the critic could not learn,
  explained_variance stuck at 0.06-0.10 and value_loss barely moving, because the
  raw reward mixes a dense bearing penalty and closing bonus with +/-100 terminal
  spikes, so the value target had huge non-stationary variance; (2) eval needed
  to load reward-normalisation stats and the --stage flag had to route to the env.
  No reward values, kinematics, sensor model, observation space, or curriculum
  logic were changed.
- config/defaults.yaml: added training.gamma 0.99. This is RecurrentPPO's existing
  default (behaviour unchanged) but is now a single config value, read once and
  shared by both RecurrentPPO and VecNormalize so the discounted-return statistic
  can never silently desync from the agent's gamma. Honours the no-hardcoding rule
  rather than literally hardcoding 0.99 in three places.
- training/train.py: wrapped the training VecEnv in VecNormalize(norm_obs=False,
  norm_reward=True, clip_reward=10.0, gamma=gamma) before passing it to
  RecurrentPPO. norm_obs stays False (hard rule: the observation is already
  bounded and meaningful). norm_reward rescales the reward by a discounted running
  std so the critic sees a roughly unit-variance target; clip_reward bounds the
  normalised signal so a single +/-100 terminal step cannot dominate an update.
  The eval VecEnv is also wrapped in VecNormalize but with norm_reward=False and
  training=False (true rewards, no stat updates): this is REQUIRED, not optional,
  because with a VecNormalize training env SB3's EvalCallback calls
  sync_envs_normalization, which asserts the eval env is also a VecNormalize
  (verified against sb3 2.7.1 source). normalize_advantage=True and gamma=gamma
  were made explicit on RecurrentPPO (both already its defaults).
- training/train.py finally block: saves vec_normalize.pkl next to the model
  (train_env.save) before final_model.zip and before closing the envs, so the
  stats are persisted even on KeyboardInterrupt and always travel with the model.
- eval/eval.py: added build_eval_env(config, stage, seed) as the single seam that
  routes --stage into InterceptEnv, and load_obs_normalizer(model_dir, config)
  which loads a sibling vec_normalize.pkl (training=False, norm_reward=False) and
  returns its normalize_obs; run_episode applies that map to obs before predict.
  Under norm_obs=False this map is the identity, so it is functionally a no-op for
  the current configuration, but it keeps eval correct if observation
  normalisation is ever enabled and makes the load explicit per the task. When no
  vec_normalize.pkl exists (older runs) it falls back to an identity map, so eval
  stays backward compatible. Rewards are never normalised at eval time: metrics
  come from the env's true reward and info dict.
- The eval rollout deliberately stays on the single gymnasium InterceptEnv (not a
  stepped VecEnv): it threads the LSTM state and reads the rich per-step info dict
  (bearing_true, termination_reason, t), which a VecEnv's auto-reset would
  obscure. VecNormalize is used only as a stateless obs transformer.
- tests/test_env.py: added test_eval_stage_flag. It asserts parse_args defaults
  --stage to 1 and parses --stage 3 to 3, and that build_eval_env(stage=1/3)
  constructs an InterceptEnv whose curriculum_stage matches. All 20 tests pass.
- Verified end to end: a 2000-step run trained with both envs VecNormalize-wrapped
  (EvalCallback fired without the sync assertion, vec_normalize.pkl written), then
  eval loaded the stats and ran; eval also ran correctly with the pkl removed.
- KAGGLE NOTE: vec_normalize.pkl must always be saved alongside the model zips and
  downloaded together from Kaggle. best_model.zip / final_model.zip are incomplete
  without the matching vec_normalize.pkl from the same run directory; eval looks
  for it next to the model and silently falls back to identity if it is missing.

## 2026-06-03 (init.bearing_offset_max 5 -> 25 deg: trivial-task fix)
- Single config change: init.bearing_offset_max 5.0 -> 25.0 deg. Nothing else
  touched (no code, env, reward, sensor, or curriculum changes).
- Why: the interceptor spawns aimed at the intruder with a random bearing offset
  drawn from +-bearing_offset_max. At 5 deg the target starts almost dead ahead,
  and because the interceptor (78 m/s) is much faster than the intruder (50 m/s)
  and spawns pointed at it, simply flying straight (the zero action) closes the
  range and detonates on a large fraction of episodes. A zero-action policy scored
  about 38% intercepts, so the task was close to trivial and the agent was never
  forced to learn to steer from bearing. Every previous training result is
  therefore meaningless as evidence of a learned bearing-only policy: most of the
  reported intercept rate was just the geometry of an easy spawn, not control.
- Widening the offset to +-25 deg places the target well off-boresight at spawn
  (but still inside the 30 deg FOV half-angle, so it remains initially visible and
  the episode is solvable), so a straight-line policy now loses the target or
  times out and the agent must actually turn toward the bearing to intercept. This
  restores the bearing-only learning signal the benchmark is supposed to measure.
- 25 deg leaves a 5 deg margin to the FOV edge, so the spawn is in view but the
  zero-action baseline is no longer a near-free win. Re-baseline before trusting
  any new run: measure the zero-action intercept rate at the new offset and treat
  it as the floor any learned policy must clear.
- tests/test_env.py: unchanged; no test pins bearing_offset_max, all 20 pass.

## 2026-06-03 (EKF + MLP + reward v4: architecture change)
- Architecture change: LSTM replaced by EKF + MLP. The EKF estimates
  theta_dot and r_dot from noisy bearing and range measurements, making
  the observation space fully observable. This eliminates the temporal
  credit assignment problem that caused explained_variance to stall.
  Reward is now quadratic on bearing angle with a soft FOV buffer zone.
- envs/sensor.py: added RelativeEKF (state [theta, theta_dot, r, r_dot],
  constant-velocity F, measurements [theta, r], range-dependent R). SensorModel
  now owns an EKF and exposes observe_full() (5-dim EKF obs + phi_self) and
  reset_filter() (seeds the EKF from the spawn measurement). observe() (3-dim)
  is unchanged for backward compatibility. observe_full() is the new Simulink
  sensor slot. Decision: the EKF q_theta/q_r and the in-update range-noise
  fraction are read from config (ekf block, sensor.range_noise_frac) and passed
  through the constructors rather than hardcoded, honouring the no-hardcoding
  rule while keeping the supplied matrices byte-identical at the default values.
- envs/intercept_env.py: observation_space is now shape (5,) with bounds
  [-pi,-pi,0,-500,-pi] .. [pi,pi,5000,500,pi]. step() uses observe_full();
  reset() calls reset_filter() before the first observe_full(). prev_r_obs/
  current_r_obs removed (closing reward gone). The true-range attribute was
  renamed self.range -> self.current_range (matches the v4 _compute_reward
  signature); the info "range" key is unchanged. _compute_reward now reads the
  stored self.last_obs: quadratic bearing penalty -0.003*theta^2, approach
  reward 0.005*max(0,-r_dot_hat), and a quadratic FOV soft-buffer penalty
  -0.05*(|theta|-20deg)^2 outside 20 deg. Decision: the termination_reason
  assignments (omitted by the literal patch) were retained, because eval, the
  info dict, and the curriculum is_success flag all read them; dropping them
  would silently break curriculum advancement (a hard rule).
- config/defaults.yaml: new ekf block (q_theta 0.01, q_r 1.0); reward block
  replaced (bearing_penalty 0.003, approach_reward 0.005, fov_soft_limit_deg 20,
  fov_edge_penalty 0.05, closing_reward 0.0 disabled, success/fov_loss/timeout
  unchanged at 100/-100/-50); lstm_hidden_size and lstm_n_layers removed from the
  training block with the LSTM-removed comment. bearing_offset_max stays 25 deg.
- training/train.py: RecurrentPPO/MlpLstmPolicy replaced by PPO/MlpPolicy;
  EntropyDecayCallback removed (and deleted from callbacks.py, it was LSTM-era
  only). gamma stays explicit and shared with VecNormalize (norm_obs still False,
  norm_reward setup unchanged). seed retained for reproducibility.
- eval/eval.py and viz/replay.py: switched RecurrentPPO -> PPO and dropped the
  LSTM state threading so main stays runnable on the new policy (not in the
  five-change list, but required by the always-runnable rule since both load and
  roll out the trained model).
- tests/test_env.py: all obs-shape assertions updated to (5,); reward tests
  rewritten for the quadratic/approach/edge formula via a controlled self.last_obs;
  closing-reward tests removed; added test_ekf_initialises, test_ekf_updates,
  test_fov_edge_penalty, test_bearing_penalty_quadratic, and
  test_zero_action_baseline. All 23 tests pass.
- Validation: 23/23 tests pass; env smoke (reset + 10 steps) prints obs shape
  (5,) and sane EKF estimates with no crash; a 1024-step PPO run trained, saved
  vec_normalize.pkl + models, and eval loaded the PPO model and wrote
  eval_results.json end to end. The zero-action baseline at 25 deg offset is
  0/50 = 0.000 (well under the 0.20 floor), confirming the task is non-trivial.

## 2026-06-03 (reward v5: bearing-rate penalty)
- Reward change only. Kinematics, EKF, observation space, curriculum, and the
  VecNormalize setup were all left untouched. The reward now has two terms:
  bearing_rate_penalty (primary) and bearing_penalty (secondary).
    - bearing_rate_penalty (alpha = 0.667) is the primary term. A constant
      line-of-sight bearing is the signature of a lead-pursuit collision course,
      so penalising the EKF bearing-rate estimate |theta_dot| teaches the policy
      to hold the target on a fixed bearing and fly a collision course rather
      than a tail chase. The calculated alpha = 0.667 gives a 23-point cumulative
      advantage to a collision course over pure pursuit.
    - bearing_penalty (beta = 0.022) is the secondary term: a soft FOV constraint
      that keeps the target near boresight without dominating the rate term.
- config/defaults.yaml: replaced the entire reward block. New keys
  bearing_rate_penalty 0.667 and bearing_penalty 0.022; success/fov_loss/timeout
  unchanged at 100/-100/-50. Removed approach_reward, fov_soft_limit_deg, and
  fov_edge_penalty (the v4 quadratic/approach/edge shaping).
- envs/intercept_env.py: _compute_reward step reward is now
  r_step = -alpha*|theta_dot| - beta*|theta|, read straight from
  self.last_obs (theta_hat, theta_dot_hat). Terminal priority is unchanged
  (success > FOV loss > timeout). Decision: the termination_reason assignments
  (omitted by the literal patch snippet) were retained because eval, the info
  dict, and the curriculum is_success flag all read them; dropping them would
  silently break curriculum advancement (a hard rule). The __init__ reward-weight
  caches were renamed to match the new formula: self.alpha now caches
  bearing_rate_penalty (primary) and self.beta caches bearing_penalty
  (secondary); both are still read directly from config inside _compute_reward,
  so the caches are introspection-only.
- tests/test_env.py: removed test_reward_values_exact,
  test_bearing_penalty_quadratic, and test_fov_edge_penalty (all keyed to the v4
  approach/edge/quadratic shaping that no longer exists). Updated
  test_reward_constants_match_spec to the v5 constants and added an absence guard
  so the removed v4 keys cannot silently reappear. Added test_bearing_rate_reward
  (asserts the two-term decomposition, both terms non-positive, rate term
  dominant) and test_collision_course_better_than_pursuit (20 zero-action steps
  from a head-on collision course vs a tail chase; cumulative reward is strictly
  higher for the collision course). All 22 tests pass.
- Validation: 22/22 tests pass. The three termination tests still hold within
  their +-5 tolerance because the v5 step shaping near a terminal state is small
  (the EKF bearing-rate kick from a re-seeded filter is ~0.05 rad/s).

---

## Physics constants (quick reference)

| Quantity              | Value     |
|-----------------------|-----------|
| Intruder speed        | 50 m/s    |
| Interceptor speed     | 78 m/s    |
| psi_dot_max           | 45 deg/s  |
| Blast radius          | 5 m       |
| FOV half-angle        | 30 deg    |
| dt                    | 0.05 s    |
| Episode timeout       | 30 s      |
| Lock-on range         | 1000 m +- 100 m |
| Lock-on bearing offset| +-25 deg  |