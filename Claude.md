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
r_step     = -0.1 * |theta_obs|       every step
r_terminal = +100  if r < 5m          (terminated, success)
           = -100  if |theta| > 30 deg (terminated, FOV loss)
           = -50   if t > 30s          (truncated, timeout)
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