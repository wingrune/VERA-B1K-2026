from dataclasses import dataclass
import json
import logging
from pathlib import Path
import numpy as np

log = logging.getLogger(__name__)
GRIPPERS = (14, 22)
CLOSED_THRESHOLD = -0.98
MIN_CLOSED_FRAC = 0.002

def derive_rules(table, min_closed_frac=MIN_CLOSED_FRAC):
    result = {}
    for idx, name in table['task_names'].items():
        stages = table['per_task'].get(str(idx), {}).get('stages', {})
        first = []
        for side in ('left', 'right'):
            hits = sorted(int(s) for s, v in stages.items()
                          if v[f'{side}_closed'] >= min_closed_frac * max(v['frames'], 1))
            first.append(hits[0] if hits else None)
        result[name] = first
    return result

def normalize_gripper_state(state):
    result = np.array(state, dtype=np.float64, copy=True)
    for i in GRIPPERS:
        result[i] = result[i] * 20.0 - 1.0
    return result

@dataclass
class RecoveryStats:
    checked: int = 0
    fired: int = 0
    left: int = 0
    right: int = 0

class RecoveryPolicy:
    def __init__(self, rules, closed_threshold=CLOSED_THRESHOLD, enabled=True):
        self.rules = {name: tuple(first) for name, first in rules.items()}
        self.closed_threshold = closed_threshold
        self.enabled = enabled
        self.stats = RecoveryStats()

    @classmethod
    def from_file(cls, path):
        return cls(json.loads(Path(path).read_text())['first_closure'])

    def closure_expected(self, task, side, stage=None):
        first = self.rules.get(task)
        if first is None:
            return True
        threshold = first[0 if side == 'left' else 1]
        return threshold is not None and (stage is None or stage >= threshold)

    def correct(self, task, stage, state, actions):
        self.stats.checked += 1
        if not self.enabled:
            return actions, False, {}
        fixes = [float(state[idx]) < self.closed_threshold and
                 not self.closure_expected(task, side, stage)
                 for idx, side in zip(GRIPPERS, ('left', 'right'))]
        if not any(fixes):
            return actions, False, {}
        result = np.tile(np.asarray(state, dtype=actions.dtype)[:actions.shape[1]], (actions.shape[0], 1))
        for idx, fix in zip(GRIPPERS, fixes):
            if fix:
                result[:, idx] = 1.0
        self.stats.fired += 1
        self.stats.left += int(fixes[0]); self.stats.right += int(fixes[1])
        info = dict(task=task, stage=stage, left=fixes[0], right=fixes[1],
                    left_val=float(state[14]), right_val=float(state[22]))
        log.info('gripper correction: task=%s stage=%s left=%s right=%s', task, stage, *fixes)
        return result, True, info

    def reset(self):

        pass
