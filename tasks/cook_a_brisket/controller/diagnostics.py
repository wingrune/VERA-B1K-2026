"""Evaluator-side ground truth for development only. Nothing here is passed to the policy."""
import json

from omnigibson.object_states import Cooked, Inside, OnTop, Open, Temperature, ToggledOn


class Diagnostics:
    def __init__(self, env, path):
        self.env, self.path = env, path
        scope = env.task.object_scope
        self.robot = env.robots[0]
        self.brisket, self.oven, self.fridge = scope["brisket.n.01_1"], scope["oven.n.01_1"], scope["electric_refrigerator.n.01_1"]
        self.board, self.pan, self.counter = scope["chopping_board.n.01_1"], scope["frying_pan.n.01_1"], scope["countertop.n.01_1"]
        self.file = open(path, "a")

    def log(self, step):
        pos = lambda o: [round(float(v), 3) for v in o.get_position_orientation()[0]]
        rp, rq = self.robot.get_position_orientation()
        row = {"step": step, "robot_pos": [round(float(v), 3) for v in rp], "robot_quat": [round(float(v), 4) for v in rq], "brisket": pos(self.brisket),
               "T": round(float(self.brisket.states[Temperature].get_value()), 1), "cooked": bool(self.brisket.states[Cooked].get_value()),
               "in_oven": bool(self.brisket.states[Inside].get_value(self.oven)), "on_board": bool(self.brisket.states[OnTop].get_value(self.board)),
               "pan_on_counter": bool(self.pan.states[OnTop].get_value(self.counter)), "oven_open": bool(self.oven.states[Open].get_value()),
               "oven_on": bool(self.oven.states[ToggledOn].get_value()), "oven_door": round(float(self.oven.joints["j_door"].get_state()[0][0]), 3),
               "fridge_doors": [round(float(self.fridge.joints[j].get_state()[0][0]), 3) for j in ("j_right_door", "j_left_door")],
               "ag": {arm: getattr(self.robot._ag_obj_in_hand[arm], "name", None) for arm in ("left", "right")}, "success": bool(self.env.task.success)}
        self.file.write(json.dumps(row) + "\n")
        self.file.flush()
