"""Entry point for b1k_evo_submit's b1k_classical backend and the evaluator: observations arrive with the env batch dimension
(--num-envs 1), the controller reads one environment's observation. The same adapter served the scored v3.9.3 runs
(runner393/unbatch.py through the official WebsocketPolicyServer)."""
from easter_policy import EasterPolicy


class Policy(EasterPolicy):
    def __init__(self):
        super().__init__()
        self.reset()                                                                                 # the backend's Policy() starts a fresh episode

    def forward(self, obs):
        return super().forward({k: v[0] for k, v in obs.items()})                                 # [1, 23]
