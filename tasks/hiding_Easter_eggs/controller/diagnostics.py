"""Evaluator-side ground truth for development only. Nothing here is passed to the policy."""
import json

import numpy as np
from omnigibson.object_states import Inside, NextTo, OnTop


class Diagnostics:
    def __init__(self, env, path):
        self.env, self.path = env, path
        scope = env.task.object_scope
        self.robot = env.robots[0]
        self.basket = scope["wicker_basket.n.01_1"]
        self.lawn = scope["lawn.n.01_1"]
        self.eggs = [scope[f"easter_egg.n.01_{i}"] for i in (1, 2, 3)]
        self.trees = {k: v for k, v in scope.items() if k.startswith("tree.n.01") and v is not None}
        self.file = open(path, "a")

    def log(self, step):
        pos = lambda o: o.get_position_orientation()[0].tolist()
        rp, rq = self.robot.get_position_orientation()
        eggs = []
        for egg in self.eggs:
            near = [k for k, t in self.trees.items() if egg.states[NextTo].get_value(t)]
            eggs.append({"pos": pos(egg), "in_basket": bool(egg.states[Inside].get_value(self.basket)),
                         "on_lawn": bool(egg.states[OnTop].get_value(self.lawn)), "next_to": near})
        row = {"step": step, "robot_pos": rp.tolist(), "robot_quat": rq.tolist(), "basket": pos(self.basket), "eggs": eggs,
               "ag": {arm: getattr(self.robot._ag_obj_in_hand[arm], "name", None) for arm in ("left", "right")},
               "reef": self.robot.eef_links["right"].get_position_orientation()[0].tolist(), "success": bool(self.env.task.success)}
        self.file.write(json.dumps(row) + "\n")
        self.file.flush()
