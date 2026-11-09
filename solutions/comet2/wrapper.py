import json
from pathlib import Path
import numpy as np
from .config import HERE, PolicyConfig
from .preprocessing import ObservationProcessor, PROPRIO_KEY, extract_state_from_proprio
from .recovery import RecoveryPolicy, normalize_gripper_state
from .scheduler import ActionScheduler

class B1KPolicyWrapper:
    def __init__(self, policy, config=None):
        self.policy = policy
        self.config = config or PolicyConfig()
        self.processor = ObservationProcessor(self.config.image_size)
        self.recovery = RecoveryPolicy.from_file(self.config.recovery_rules)
        self.scheduler = ActionScheduler(self.config)
        self.task_ids = json.loads((HERE / 'assets' / 'task_ids.json').read_text())
        self.task_name = ''
        self.text_prompt = ''

    def reset(self):
        self.scheduler.reset()
        self.recovery.reset()


        self.policy.reset()

    def set_task(self, task_name):
        if task_name and task_name not in self.task_ids.values():
            raise KeyError(task_name)
        if self.task_name != task_name:
            self.task_name = self.text_prompt = task_name
            self.reset()

    def process_obs(self, obs):
        return self.processor(obs, self.text_prompt)

    def act_chunk(self, obs, count):
        task = self.task_ids.get(str(int(np.asarray(obs['task_id']).reshape(-1)[0])), '')
        self.set_task(task)

        model_obs = self.process_obs(obs)
        if self.scheduler.needs_inference(count):
            actions = self.policy.infer(model_obs)['actions'].copy()
            state = normalize_gripper_state(extract_state_from_proprio(model_obs['observation/state']))
            actions, _, _ = self.recovery.correct(self.task_name, None, state, actions)
            self.scheduler.plan(actions)
        result = self.scheduler.take(count)
        return result[None, ...] if np.asarray(obs[PROPRIO_KEY]).ndim != 1 else result

    def act(self, obs):
        return self.act_chunk(obs, 1)[..., 0, :]
