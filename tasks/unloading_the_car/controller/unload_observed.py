"""Unloading the car from what the robot's own cameras show: the estimator and the entry point of the script.

The base pose, the pose fixes behind the car and the per-step runtime are those of the controller for loading the
car (car_observed.py). What is new here is what is known of the two bags: where each lies in the trunk, from the
head camera's depth image (trunk_view.py).
"""
import numpy as np
from manip import pose2, BASE_Z
from car_observed import ObservedEstimator, localize
from trunk_view import Bags
from unload_pipeline import full_task


class BagEstimator(ObservedEstimator):
    def __init__(self, observer, scene, folder):
        super().__init__(observer, scene)
        self.shapes = Bags(folder); self.bag = {}           # name -> what is known of the bag (trunk_view.Bags.see), in M

    # (Nothing stands about on the living-room floor in this task, and nothing is looked for on the table.)
    def see(self, what=()):
        return {}

    def where(self, name):
        return None

    def look_into_trunk(self):
        """Take what the head camera shows of the bags in the open trunk. Returns {name: what is known or None} of
        this look (see trunk_view.Bags.see: origin and turn of the bag's frame in M)."""
        points = self.ob.points('head', 2); world = pose2(*self.base()); in_map = points @ world[:3, :3].T + world[:3, 3]; in_map[:, 2] = points[:, 2] + BASE_Z
        found, count = self.shapes.see(in_map)
        for name, known in found.items():
            if known is not None:
                self.bag[name] = dict(known, step=self.ob.steps)
        return found

    def bag_in_base(self, name):
        """(origin, turn) of a bag's frame in the base frame, by the present pose estimate."""
        known = self.bag[name]; world = pose2(*self.base()); inverse = np.linalg.inv(world)
        origin = inverse[:3, :3] @ known['origin'] + inverse[:3, 3]; origin[2] = known['origin'][2] - BASE_Z
        return origin, inverse[:3, :3] @ known['turn']


def observed_pipeline(rt):
    located = yield from localize(rt)
    if not located:
        return 'not_localized'
    return (yield from full_task(rt.ctx, rt.est))
