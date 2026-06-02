"""Curriculum scheduler (spec section 4.7).

Advances the intruder difficulty stage when EITHER condition holds, evaluated
over the most recent window of episodes:
  (a) rolling intercept rate exceeds this stage's threshold, or
  (b) the rolling intercept rate has not set a new best for `patience` episodes
      (a plateau fallback so a stuck agent still progresses).

The scheduler only owns the advancement logic and the rolling statistics. Wiring
the stage into the running environments is the callback's job (spec: stage is
pushed to InterceptEnv via a VecEnv attribute update).
"""
from __future__ import annotations

from collections import deque

# Tiny margin so floating-point equality does not count as an improvement.
_IMPROVE_EPS = 1e-9


class CurriculumScheduler:
    """Tracks intercept rate over a sliding window and advances the stage."""

    def __init__(
        self,
        thresholds: list[float | None] = (0.70, 0.65, None),
        window_size: int = 100,
        patience: int = 500,
    ) -> None:
        """
        Args:
            thresholds: per-stage intercept-rate threshold. None marks the final
                stage (never advances on condition (a)). Length defines the
                number of stages; stages are numbered 1..N.
            window_size: number of recent episodes in the rolling window.
            patience: episodes without a new best rolling rate before advancing.
        """
        self.thresholds = list(thresholds)
        self.stages = list(range(1, len(self.thresholds) + 1))
        self.window_size = int(window_size)
        self.patience = int(patience)

        self._stage_idx = 0
        self._window: deque[bool] = deque(maxlen=self.window_size)
        self._best_rate = 0.0
        self._since_improve = 0

    @property
    def current_stage(self) -> int:
        """Current curriculum stage (1-based)."""
        return self.stages[self._stage_idx]

    @property
    def intercept_rate(self) -> float:
        """Rolling intercept rate over the current window (0.0 if empty)."""
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    @property
    def is_final_stage(self) -> bool:
        return self._stage_idx >= len(self.stages) - 1

    def update(self, episode_result: bool) -> bool:
        """Record one episode outcome (True = intercept) and maybe advance.

        Returns:
            True if the stage advanced on this call, else False.
        """
        self._window.append(bool(episode_result))

        # Final stage never advances; keep recording so intercept_rate stays live.
        if self.is_final_stage:
            return False

        # Need a full window before the rate is meaningful.
        if len(self._window) < self.window_size:
            return False

        rate = self.intercept_rate

        # Plateau tracking for condition (b).
        if rate > self._best_rate + _IMPROVE_EPS:
            self._best_rate = rate
            self._since_improve = 0
        else:
            self._since_improve += 1

        threshold = self.thresholds[self._stage_idx]
        meets_threshold = threshold is not None and rate > threshold
        plateaued = self._since_improve >= self.patience

        if meets_threshold or plateaued:
            self._advance()
            return True
        return False

    def _advance(self) -> None:
        """Move to the next stage and reset the rolling statistics."""
        if self.is_final_stage:
            return
        self._stage_idx += 1
        self._window.clear()
        self._best_rate = 0.0
        self._since_improve = 0
