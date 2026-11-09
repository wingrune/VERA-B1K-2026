from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

LEFT_GRIPPER_IDX = 14
RIGHT_GRIPPER_IDX = 22
CLOSED_THRESHOLD = -0.98
OPEN_COMMAND = 1.0


MIN_CLOSED_FRAC = 0.002


@dataclasses.dataclass
class CorrectionStats:
    checked: int = 0
    fired: int = 0
    left: int = 0
    right: int = 0


class GripperCorrector:


    def __init__(self, table_path: str | Path, closed_threshold: float = CLOSED_THRESHOLD,
                 min_closed_frac: float = MIN_CLOSED_FRAC, enabled: bool = True):
        self.enabled = enabled
        self.closed_threshold = closed_threshold
        self.min_closed_frac = min_closed_frac
        self.stats = CorrectionStats()
        t = json.loads(Path(table_path).read_text())
        names = t["task_names"]
        self._first_closure: dict[str, dict[str, int | None]] = {}

        def _first(per, side):


            hits = sorted(int(s) for s, v in per.items()
                          if v[f"{side}_closed"] >= self.min_closed_frac * max(v["frames"], 1))
            return hits[0] if hits else None

        for idx, name in names.items():
            per = t["per_task"].get(str(idx), {}).get("stages", {})
            left = [_first(per, "left")] if _first(per, "left") is not None else []
            right = [_first(per, "right")] if _first(per, "right") is not None else []

            self._first_closure[name] = {"left": left[0] if left else None,
                                         "right": right[0] if right else None}
        logger.info("GripperCorrector: %d tasks; never-close left=%d right=%d",
                    len(self._first_closure),
                    sum(1 for v in self._first_closure.values() if v["left"] is None),
                    sum(1 for v in self._first_closure.values() if v["right"] is None))

    def _closure_expected(self, task: str, side: str, stage: int | None) -> bool:

        rec = self._first_closure.get(task)
        if rec is None:
            return True
        first = rec[side]
        if first is None:
            return False
        if stage is None:
            return True
        return stage >= first

    def correct(self, task: str, stage: int | None, state: np.ndarray, actions: np.ndarray):

        self.stats.checked += 1
        if not self.enabled:
            return actions, False, {}
        left_closed = float(state[LEFT_GRIPPER_IDX]) < self.closed_threshold
        right_closed = float(state[RIGHT_GRIPPER_IDX]) < self.closed_threshold
        fix_left = left_closed and not self._closure_expected(task, "left", stage)
        fix_right = right_closed and not self._closure_expected(task, "right", stage)
        if not (fix_left or fix_right):
            return actions, False, {}

        corrected = np.tile(np.asarray(state, dtype=actions.dtype)[: actions.shape[1]], (actions.shape[0], 1))
        if fix_left:
            corrected[:, LEFT_GRIPPER_IDX] = OPEN_COMMAND
            self.stats.left += 1
        if fix_right:
            corrected[:, RIGHT_GRIPPER_IDX] = OPEN_COMMAND
            self.stats.right += 1
        self.stats.fired += 1
        info = {"task": task, "stage": stage, "left": fix_left, "right": fix_right,
                "left_val": float(state[LEFT_GRIPPER_IDX]), "right_val": float(state[RIGHT_GRIPPER_IDX])}
        logger.info("gripper correction: task=%s stage=%s left=%s right=%s", task, stage, fix_left, fix_right)
        return corrected, True, info


def normalize_gripper_state(state23: np.ndarray) -> np.ndarray:

    out = np.array(state23, dtype=np.float64, copy=True)
    for i in (LEFT_GRIPPER_IDX, RIGHT_GRIPPER_IDX):
        out[i] = out[i] * 20.0 - 1.0
    return out


def stage_of(frame: int, episode_length: int, num_stages: int) -> int:

    if episode_length <= 0 or num_stages <= 0:
        return 0
    return int(np.clip(int(frame / (episode_length / num_stages)), 0, num_stages - 1))


BASE_VELOCITY_SLICE = slice(0, 3)
GRIPPER_VARIATION_THRESHOLD = 0.2


def check_gripper_variation(actions: np.ndarray, n_check: int) -> tuple[bool, float, float]:

    w = actions[:n_check]
    left = float(np.max(w[:, LEFT_GRIPPER_IDX]) - np.min(w[:, LEFT_GRIPPER_IDX]))
    right = float(np.max(w[:, RIGHT_GRIPPER_IDX]) - np.min(w[:, RIGHT_GRIPPER_IDX]))
    return (left > GRIPPER_VARIATION_THRESHOLD or right > GRIPPER_VARIATION_THRESHOLD), left, right


def compress_action_chunk(actions: np.ndarray, n_from: int, n_to: int) -> tuple[np.ndarray, bool]:

    from scipy.interpolate import interp1d

    actions = np.asarray(actions)

    def _uncompressed() -> np.ndarray:

        out = actions[:n_to].copy()
        if out.shape[0] < n_to:
            pad = np.repeat(out[-1:], n_to - out.shape[0], axis=0)
            pad[:, BASE_VELOCITY_SLICE] = 0.0
            out = np.concatenate([out, pad], axis=0)
        return out

    if n_to >= n_from or actions.shape[0] < n_from or n_from < 4:
        return _uncompressed(), False
    high_var, _, _ = check_gripper_variation(actions, n_from)
    if high_var:
        return _uncompressed(), False

    src = actions[:n_from]
    xs = np.linspace(0, n_from - 1, n_from)
    xt = np.linspace(0, n_from - 1, n_to)
    out = np.empty((n_to, src.shape[1]), dtype=np.float64)
    for d in range(src.shape[1]):
        out[:, d] = interp1d(xs, src[:, d], kind="cubic")(xt)


    out[:, BASE_VELOCITY_SLICE] *= (n_from - 1) / (n_to - 1)
    return out.astype(actions.dtype), True
