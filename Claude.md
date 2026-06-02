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
│   └── sensor.py               # SensorModel: bearing + range + FOV gate
├── training/
│   ├── train.py                # PPO+LSTM entry point
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

### Observation vector — shape (3,) float32
| Index | Variable  | Units | Noise                    |
|-------|-----------|-------|--------------------------|
| 0     | theta_obs | rad   | N(0, 0.5 deg) additive   |
| 1     | r_obs     | m     | N(0, 0.05 * r) scaled    |
| 2     | phi_self  | rad   | noiseless                |

### Action vector — shape (1,) float32, bounds [-1, 1]
| Index | Variable   | Physical meaning            |
|-------|------------|-----------------------------|
| 0     | turn_rate  | scaled to psi_dot_max deg/s |

To add thrust later: append index 1 in ActionConfig and one line in
Interceptor.apply_action(). Nothing else changes.

### SensorModel.observe() signature
```python
def observe(
    self,
    interceptor_state: dict,
    intruder_state: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    # returns (obs_3dim, in_fov)
```

### Reward
```
r_step     = -0.05 * |theta_obs|                       every step
           + 0.015 * max(0, r_obs[t-1] - r_obs[t])    closing reward
r_terminal = +100  if r < 5m                           (terminated, success)
           = -100  if |theta| > 30 deg                 (terminated, FOV loss)
           = -100  if t > 30s                          (truncated, timeout)
```

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
| Lock-on bearing offset| +-5 deg   |