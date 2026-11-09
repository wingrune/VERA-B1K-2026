"""Fair planner of task 98 (installing_a_scanner): only the native observation (3 RGB-D cameras + proprio) comes in, the 23D action goes out.

Plan: turn in place until the desk, the laptop and the scanner with its button end are perceived (scan), walk to the front of the scanner and
press the button once with a fingertip (plan_button, go_button, reach_button, press_button, retreat_button), then pinch a top edge of the scanner
(the native assisted grasp holds it), lift it, carry it next to the laptop and put it down (plan_grasp, go_grasp, pre_grasp, reach_grasp,
close_grasp, back_off, lift, plan_carry, carry, pre_place, lower, release, retreat_grasp), then hold still (done). When no pinch is reachable
the scanner is slid to the laptop with two open fingertips instead (plan_push, go_push, reach_push, push, retreat_push). Every step is a named phase
with a reason for each transition, listed by log() for the trace of the harness. IK on the GPU (cuRobo worker, ik.py): reachable_stands
for the base poses, solve_near for the arm servo. Interface: Planner.reset(seed), Planner.act(obs) -> (1, 23)
float32, Planner.log() -> dict; entrypoint() builds the planner.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.skills_lib import r1pro_kinematics as K

from .phases import approach_button, carry_place, find_scanner, grasp_scanner, press_button, push_to_laptop, recover
from . import ik
from . import params as P
from .phases.base import Core

PHASES = {**find_scanner.PHASES, **approach_button.PHASES, **press_button.PHASES, **grasp_scanner.PHASES, **carry_place.PHASES,
          **push_to_laptop.PHASES, **recover.PHASES}
EST_EVERY = 30
HOLD_PHASES = ('reach_button', 'press_button', 'retreat_button', 'reach_push', 'push', 'retreat_push', 'pre_grasp', 'reach_grasp', 'close_grasp',
               'back_off', 'lift', 'pre_place', 'lower', 'release', 'retreat_grasp')
PUSH_PHASES = ('plan_push', 'go_push', 'reach_push', 'push', 'retreat_push', 'plan_grasp', 'go_grasp')


class Planner(Core):
    def __init__(self) -> None:
        super().__init__()
        self.est_cache: Dict[str, Any] = {}

    def reset(self, seed: int = 0) -> None:
        self.seed = seed
        ik.start()                       # GPU IK worker: once per process, warms up while the episode loads and settles
        self.reset_state()
        self.est_cache = {}

    def act(self, obs: Dict[str, Any], privileged: Any = None) -> np.ndarray:
        assert privileged is None, 'the fair planner takes the observation only'
        self.perc.allow_partial = self.phase in PUSH_PHASES
        self.perc.freeze_pose = self.phase in HOLD_PHASES
        self.perc.observe(obs, self.step)
        world = self.current_world()
        q = K.q_from_proprio(self.perc.proprio)
        if self.cmd is None:
            self.cmd = np.r_[np.zeros(3), q[0:4], q[4:11], -1., q[13:20], -1.]
        action = self.cmd.copy()
        action[:3] = 0.
        action[14], action[22] = self.grip['left'], self.grip['right']
        self.step += 1
        self.age += 1
        if self.perc.lift > P.LIFT_MAX_M and self.phase in recover.NAV_PHASES:
            recover.start_unstick(self)
        PHASES[self.phase](self, action, q, world)
        self.cmd = action.copy()
        assert np.isfinite(action).all()
        return action[None].astype(np.float32)

    def current_world(self) -> Any:
        """The World the phases plan on: the perceived one (a hook for the oracle-swap tool, which substitutes the truth)."""
        return self.perc.world()

    def log(self) -> Dict[str, Any]:
        if self.step % EST_EVERY == 1 or not self.est_cache:
            self.est_cache = self.perc.est()
        return dict(phase=self.phase, reason=self.reason, step=self.step, age=self.age, side=self.side, tilt=self.tilt,
                    err=[round(float(v), 4) for v in self.err], counts=dict(self.counts), push_p=round(self.push_p, 3),
                    press_s=round(self.press_s, 4), est=self.est_cache, ik=ik.stats(), history=self.history[-12:],
                    pick={k: self.pick[k] for k in ('tries', 'fails', 'drops', 'place_rounds', 'placed')})


def entrypoint() -> Planner:
    return Planner()
