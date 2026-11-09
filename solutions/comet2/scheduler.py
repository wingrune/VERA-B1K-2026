from dataclasses import dataclass
import numpy as np
from scipy.interpolate import interp1d

def compress_action_chunk(actions, n_from=26, n_to=20):
    actions = np.asarray(actions)
    if len(actions) == 0:
        raise ValueError('Empty model action chunk')
    window = actions[:n_from]
    high_variation = any(float(np.max(window[:, i]) - np.min(window[:, i])) > 0.2 for i in (14, 22))
    if n_to >= n_from or len(actions) < n_from or n_from < 4 or high_variation:
        result = actions[:n_to].copy()
        if len(result) < n_to:
            pad = np.repeat(result[-1:], n_to - len(result), axis=0)
            pad[:, :3] = 0.0
            result = np.concatenate([result, pad], axis=0)
        return result, False
    source = actions[:n_from]
    xs = np.linspace(0, n_from - 1, n_from)
    xt = np.linspace(0, n_from - 1, n_to)
    result = np.empty((n_to, source.shape[1]), dtype=np.float64)
    for dim in range(source.shape[1]):
        result[:, dim] = interp1d(xs, source[:, dim], kind='cubic')(xt)
    result[:, :3] *= (n_from - 1) / (n_to - 1)
    return result.astype(actions.dtype), True

@dataclass
class ActionQueue:
    actions: np.ndarray | None = None
    index: int = 0

    @property
    def remaining(self):
        return 0 if self.actions is None else len(self.actions) - self.index

    def replace(self, actions):

        self.actions = np.array(actions, dtype=np.float32, copy=True)
        self.index = 0

    def pop(self, count):
        if count > self.remaining:
            raise ValueError('Action queue underflow')
        result = self.actions[self.index:self.index + count].copy()
        self.index += count
        return result

class ActionScheduler:
    def __init__(self, config):
        self.config = config
        self.n_chunks = 0
        self.n_compressed = 0
        self.reset()

    def reset(self):
        self.queue = ActionQueue()
        self.step_counter = 0

    def needs_inference(self, count):
        if not 1 <= count <= self.config.execute_steps:
            raise ValueError(f'Chunk size must be 1..{self.config.execute_steps}')

        if self.config.execute_steps - self.step_counter % self.config.execute_steps < count:
            self.step_counter = 0
        return self.queue.remaining == 0 or self.step_counter % self.config.execute_steps == 0

    def plan(self, actions):
        actions = np.asarray(actions)
        if actions.ndim != 2 or actions.shape[1] != self.config.action_dim or not np.isfinite(actions).all():
            raise ValueError('Expected finite model actions (T,23)')
        processed, compressed = compress_action_chunk(actions, self.config.compression_from, self.config.execute_steps)
        self.queue.replace(processed)
        self.n_chunks += 1
        self.n_compressed += int(compressed)

    def take(self, count):
        result = self.queue.pop(count)
        self.step_counter += count
        return result
