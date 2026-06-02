# Intercepterp — Full Project Specification

> Counter-UAS interception via bearing-only RL in a GPS-denied environment.
> Named after the Terp. Obviously.

---

## 1. Project summary

A 2D pursuit-evasion environment where a fixed-wing interceptor learns to
destroy a Shahed-136 class intruder using only bearing angle and noisy range
as observations — replicating the sensor output of a monocular camera in a
GPS-denied scenario. The trained policy is the middle block in a modular
pipeline designed for future Simulink integration.

---

## 2. Repo layout

```
intercepterp/
├── CLAUDE.md
├── README.md
├── config/
│   └── defaults.yaml
├── envs/
│   ├── base_env.py
│   ├── intercept_env.py
│   ├── agents/
│   │   ├── intruder.py
│   │   └── interceptor.py
│   └── sensor.py
├── training/
│   ├── train.py
│   ├── curriculum.py
│   └── callbacks.py
├── eval/
│   ├── eval.py
│   └── metrics.py
├── viz/
│   ├── dashboard.py
│   └── replay.py
├── tests/
│   └── test_env.py
└── runs/          # gitignored
```

---

## 3. Physics parameters

All values grounded in open-source hardware references.

| Parameter              | Value       | Source                                      |
|------------------------|-------------|---------------------------------------------|
| Intruder speed         | 50 m/s      | Shahed-136 ~180 km/h (CSIS Missile Threat)  |
| Interceptor speed      | 78 m/s      | Shvidun-class 280 km/h                      |
| Interceptor ψ̇_max     | 45 °/s      | Conservative fixed-wing bound               |
| Blast radius           | 5 m         | Small kinetic + fragmentation charge        |
| FOV half-angle         | 30°         | Forward fixed camera with zoom              |
| Lock-on range          | 1000 ± 100 m | Operational scenario                       |
| Lock-on bearing offset | ±5°         | Operator pre-alignment                      |
| Episode timeout        | 30 s        | Geometric bound at these speeds             |
| dt                     | 0.05 s      | 20 Hz control loop                         |

---

## 4. Class specifications

### 4.1 `ActionConfig` (dataclass, `config/defaults.yaml` + Python mirror)

Single source of truth for the action space. Adding thrust later means adding
one field here and one line in `Interceptor.apply_action`.

```python
@dataclass
class ActionConfig:
    # v1: turn rate only
    turn_rate: bool = True        # ψ̇ ∈ [−ψ̇_max, +ψ̇_max]

    # v2 placeholder: uncomment to enable
    # thrust: bool = False        # dv ∈ [−a_max, +a_max]  m/s²

    @property
    def dim(self) -> int:
        return sum([self.turn_rate])  # extend: + self.thrust
```

The `InterceptEnv` reads `action_config.dim` to define `action_space`. No
other file needs to know the action dimension.

---

### 4.2 `BaseEnv` (`envs/base_env.py`)

Abstract base. Enforces the interface every env variant must satisfy.

```python
class BaseEnv(gymnasium.Env, ABC):
    @abstractmethod
    def _get_obs(self) -> np.ndarray: ...

    @abstractmethod
    def _get_info(self) -> dict: ...

    @abstractmethod
    def _compute_reward(self) -> tuple[float, bool, bool]: ...
    # returns: (reward, terminated, truncated)
```

---

### 4.3 `Intruder` (`envs/agents/intruder.py`)

Dubins kinematics. Evasion behaviour is set by curriculum stage.

```python
class Intruder:
    speed: float = 50.0           # m/s, fixed
    psi_dot_max: float = 15.0     # °/s, max evasive turn rate

    def __init__(self, stage: int): ...

    def reset(self, rng: np.random.Generator) -> None:
        # Place intruder at origin facing a random heading.
        # Interceptor is placed relative to it.

    def step(self, dt: float) -> None:
        # Stage 1: straight flight, no evasion
        # Stage 2: sinusoidal heading oscillation, period ~8s
        # Stage 3: reactive evasion — turns away from interceptor bearing
        #           (requires knowing interceptor bearing, passed in as arg)

    def get_state(self) -> dict:
        return {"x": ..., "y": ..., "psi": ..., "speed": ...}
```

---

### 4.4 `Interceptor` (`envs/agents/interceptor.py`)

Dubins kinematics. Action unpacking is index-based for easy extension.

```python
class Interceptor:
    speed: float = 78.0           # m/s, fixed (v1)
    psi_dot_max: float = 45.0     # °/s

    def reset(self, intruder_state: dict, rng: np.random.Generator) -> None:
        # Place interceptor at range ~ U(900, 1100) m from intruder
        # at bearing offset ~ U(-5°, +5°) from intruder heading
        # with own heading pointing roughly toward intruder

    def apply_action(self, action: np.ndarray) -> None:
        # index 0: turn rate command, clipped to ±psi_dot_max
        # index 1 (future): thrust command, clipped to ±a_max
        psi_dot_cmd = np.clip(action[0], -self.psi_dot_max, self.psi_dot_max)
        self.psi += psi_dot_cmd * dt
        self.x   += self.speed * np.cos(self.psi) * dt
        self.y   += self.speed * np.sin(self.psi) * dt
        # When thrust is added: self.speed += thrust_cmd * dt

    def get_state(self) -> dict:
        return {"x": ..., "y": ..., "psi": ..., "speed": ...}
```

---

### 4.5 `SensorModel` (`envs/sensor.py`)

The exact interface between simulation truth and policy input.
This is the block that gets replaced by a YOLO pipeline in Simulink.

```python
class SensorModel:
    bearing_noise_std: float = 0.5   # degrees
    range_noise_frac:  float = 0.05  # 5% of true range
    fov_half_angle:    float = 30.0  # degrees

    def observe(
        self,
        interceptor_state: dict,
        intruder_state: dict,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, bool]:
        """
        Returns:
            obs: np.ndarray shape (3,)
                [0] theta_obs  — bearing angle to target, radians, noisy
                [1] r_obs      — range to target, metres, noisy
                [2] phi_self   — own heading, radians, noiseless
            in_fov: bool — False triggers episode termination
        """

    def _bearing(self, interceptor: dict, intruder: dict) -> float:
        # Angle from interceptor body x-axis to intruder, in [-pi, pi]
        dx = intruder["x"] - interceptor["x"]
        dy = intruder["y"] - interceptor["y"]
        bearing_world = np.arctan2(dy, dx)
        return wrap_to_pi(bearing_world - interceptor["psi"])

    def _range(self, interceptor: dict, intruder: dict) -> float:
        dx = intruder["x"] - interceptor["x"]
        dy = intruder["y"] - interceptor["y"]
        return np.sqrt(dx**2 + dy**2)
```

The observation vector is always shape `(3,)` in v1. When a 4th dimension
(e.g. range rate) is added, it goes here and `InterceptEnv.observation_space`
updates accordingly — nowhere else.

---

### 4.6 `InterceptEnv` (`envs/intercept_env.py`)

The main Gymnasium environment.

```python
class InterceptEnv(BaseEnv):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: dict,                  # loaded from defaults.yaml
        action_config: ActionConfig,
        curriculum_stage: int = 1,
        render_mode: str | None = None,
        rng_seed: int | None = None,
    ): ...

    # --- Gymnasium API ---

    observation_space: gymnasium.spaces.Box
    # shape (3,), dtype float32
    # bounds: theta_obs ∈ [-pi, pi], r_obs ∈ [0, 5000], phi_self ∈ [-pi, pi]

    action_space: gymnasium.spaces.Box
    # shape (action_config.dim,), dtype float32, bounds [-1, 1]
    # policy outputs in [-1, 1], env scales to physical units internally
    # reason: keeps action space stable when thrust is added

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]: ...

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # 1. apply_action to interceptor
        # 2. step intruder (pass interceptor bearing for stage-3 evasion)
        # 3. observe via SensorModel
        # 4. compute reward
        # 5. check termination
        # returns: obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        # rgb_array: matplotlib top-down view
        # interceptor heading shown as arrow
        # intruder shown as triangle
        # bearing cone drawn as shaded wedge (FOV)
        # used by replay.py only, not during training

    # --- Internal ---

    def _compute_reward(self) -> tuple[float, bool, bool]:
        r_step = -0.1 * abs(self.sensor.last_bearing)

        if self.range < self.config["blast_radius"]:
            return r_step + 100.0, True, False   # terminated: success

        if not self.in_fov:
            return r_step - 100.0, True, False   # terminated: FOV loss

        if self.t >= self.config["episode_timeout"]:
            return r_step - 50.0, False, True    # truncated: timeout

        return r_step, False, False
```

**Important convention:** `terminated=True` means the episode ended due to an
environment event (success or FOV loss). `truncated=True` means it hit the
time limit. SB3 handles these correctly for GAE computation.

---

### 4.7 `CurriculumScheduler` (`training/curriculum.py`)

```python
class CurriculumScheduler:
    """
    Advances stage when EITHER condition is met:
      (a) intercept_rate > threshold over last window_size episodes
      (b) no improvement for patience episodes

    Stage is passed to InterceptEnv via a VecEnv attribute update.
    """

    stages: list[int] = [1, 2, 3]
    thresholds: list[float] = [0.70, 0.65, None]  # None = final stage
    window_size: int = 100
    patience: int = 500           # episodes with no improvement → advance

    def update(self, episode_result: bool) -> bool:
        # Returns True if stage advanced this call
        ...

    @property
    def current_stage(self) -> int: ...
```

Thresholds are intentionally lower on stage 2+ because the task is harder.
Tune these after first training run.

---

## 5. Reward function

```
r_step     = -0.1 * |θ_obs|          applied every step
r_terminal = +100   if r < 5m        (success)
           = -100   if |θ| > 30°     (FOV loss)
           = -50    if t > 30s       (timeout)
```

The bearing penalty `r_step` does two jobs:
- Keeps the target in FOV naturally (high |θ| is continuously punished)
- Shapes toward collision course (constant bearing = zero bearing rate = zero penalty)

No distance reward. The agent closes range because that is the only path to +100.
No time penalty. Add `-0.01 * dt` per step only if agent learns to dawdle.

---

## 6. Observation space

| Index | Variable   | Units   | Noise model              |
|-------|------------|---------|--------------------------|
| 0     | θ_obs      | rad     | N(0, 0.5°) additive      |
| 1     | r_obs      | m       | N(0, 0.05·r) multiplied  |
| 2     | φ_self     | rad     | noiseless (own heading)  |

Range noise is proportional to range (5%), modelling bounding-box apparent
size estimation from a monocular camera. At lock-on (~1km): ±50m. At
terminal approach (~50m): ±2.5m.

**Future extension:** add index 3 = θ̇_obs (bearing rate, finite difference
with noise) when ablation study shows it is needed. LSTM should learn to
estimate it implicitly from history — test this first.

---

## 7. Training configuration

```yaml
# config/defaults.yaml

physics:
  intruder_speed:    50.0      # m/s
  interceptor_speed: 78.0      # m/s
  psi_dot_max:       45.0      # deg/s
  blast_radius:      5.0       # m
  fov_half_angle:    30.0      # deg
  dt:                0.05      # s
  episode_timeout:   30.0      # s

init:
  range_mean:        1000.0    # m
  range_std:         100.0     # m
  bearing_offset_max: 5.0      # deg

sensor:
  bearing_noise_std: 0.5       # deg
  range_noise_frac:  0.05      # fraction of true range

reward:
  bearing_penalty:   0.1       # alpha in r_step = -alpha * |theta|
  success:           100.0
  fov_loss:         -100.0
  timeout:          -50.0

training:
  total_timesteps:   5_000_000
  n_envs:            8
  n_steps:           2048
  batch_size:        256
  n_epochs:          10
  learning_rate:     3.0e-4
  clip_range:        0.2
  lstm_hidden_size:  256
  lstm_n_layers:     1

curriculum:
  window_size:       100
  patience:          500
  thresholds:        [0.70, 0.65, null]
```

---

## 8. Viz — fast simulation loop

### `viz/dashboard.py`

Runs as a subprocess during training. Reads SB3 TensorBoard logs or a
shared JSON file written by `callbacks.py`. Shows:

- Rolling intercept rate (last 100 eps) — the primary metric
- Mean |θ| per episode — secondary metric
- Current curriculum stage (highlighted when it advances)
- Time-to-intercept histogram (updates every eval interval)

Target: everything visible in one terminal-sized matplotlib window.
Update every 10 episodes. No web server, no Streamlit — just matplotlib
in non-blocking mode (`plt.pause()`).

### `viz/replay.py`

```bash
python viz/replay.py --run runs/latest
python viz/replay.py --run runs/20260602_143200 --episode 42
```

Outputs:
- Animated top-down trajectory (interceptor + intruder paths + FOV cone)
- Bearing angle θ(t) time series
- Terminal summary: SUCCESS/FAIL, reason, t_final, r_final

### `eval/eval.py`

```bash
python eval/eval.py --model runs/latest/best_model.zip --n-eps 100
```

Outputs JSON to `runs/latest/eval_results.json`:
```json
{
  "intercept_rate": 0.87,
  "mean_time_to_intercept": 14.3,
  "mean_bearing_error": 0.08,
  "fov_loss_rate": 0.08,
  "timeout_rate": 0.05,
  "stage": 3
}
```

---

## 9. Simulink integration plan (future)

The three Simulink interface points are:

1. **Plant slot** (dashed boundary, simulation layer): replace `Intruder` and
   `Interceptor` with a Simulink co-simulation block. The block receives
   `ψ̇_cmd` and returns `[x_int, y_int, ψ_int, x_tgt, y_tgt]`. The
   `SensorModel` runs unchanged on top.

2. **Sensor slot** (dashed boundary, sensor layer): replace `SensorModel.observe()`
   with a call that reads `[θ, r]` from a Simulink YOLO output block. The 3-dim
   obs vector is identical.

3. **Policy export**: trained policy is exported as ONNX. Simulink calls it via
   a MATLAB function block wrapping `onnxruntime`. SB3 RecurrentPPO supports
   ONNX export; LSTM hidden state is passed as an additional input/output pair.

None of these changes touch the reward function, curriculum, or training code.

---

## 10. Conventions

- No em dashes anywhere in code, comments, or docs.
- All physical units in SI (metres, seconds, radians) internally.
  Degrees only in config yaml and printed output.
- `rng: np.random.Generator` passed explicitly everywhere — no global random
  state. Seed from `InterceptEnv.reset(seed=...)`.
- Git: `main` branch is always runnable. Feature branches named
  `feature/description`. Conventional commits.
- Runs saved to `runs/YYYYMMDD_HHMMSS/`. Contains: config snapshot,
  best_model.zip, final_model.zip, eval_results.json, tensorboard/.
- `runs/` is gitignored. Models are not committed.
- CLAUDE.md contains: project summary, file map, session log template,
  and a reminder that the Simulink interface points must never be broken.