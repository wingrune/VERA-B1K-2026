"""Sweeping the garage from what the robot's own cameras show: the estimator and the entry point.

The base pose and the per-step runtime are those of the earlier controllers (car_observed.py), with the map of the
garage. What is new here is what is known of the broom (broom_view.py): where it lies, which way, where its handle is
taken. The sand and the dust are not looked for: the floor is swept by a fixed plan (sweep.py).
"""
import numpy as np
from manip import TALL_TRUNK
from car_observed import ObservedEstimator, localize
from broom_view import BroomView
from observer import WRIST, SLOT, FOCAL
from localizer import camera_points, BASE_Z

NEAR = 2.5                          # a look from nearer than this replaces what was known from further away


class BroomEstimator(ObservedEstimator):
    def __init__(self, observer, scene, folder):
        super().__init__(observer, scene)
        self.view = BroomView(scene); self.broom = None; self.looks = 0; self.switched = None
        self.held = set()

    def see(self, what=()):
        return {}

    def predict(self, s, last_action, dt=1 / 30):
        super().predict(s, last_action, dt)
        switched = self.ob.loc.debug.get('switched_to') if getattr(self.ob.loc, 'debug', None) else None
        if switched is not None and switched != self.switched:
            self.news.append(dict(start_pose_switched_to=switched))
        self.switched = switched
        if self.ob.loc.pose is not None and self.ob.steps % 6 == 0 and not getattr(self, 'holding_broom', False) and abs(float(last_action[2])) < .3 and float(np.max(np.abs(np.asarray(s, float)[53:57] - TALL_TRUNK))) < .06:
            self.look()

    def look(self):
        base = self.base(); self.looks += 1
        seen = self.view.broom(self.ob, base)
        if seen is None or seen['cut']:
            return seen
        old = self.broom
        if old is None or seen['range'] < max(old['range'], NEAR) or float(np.linalg.norm(seen['head'] - old['head'])) > .25:
            if old is None:
                self.news.append(dict(seen='broom', head=np.round(seen['head'], 3).tolist(), range=round(seen['range'], 2)))
            self.broom = dict(seen, step=self.ob.steps)
        return seen

    def wrist_handle(self, arm, tip, direction):
        """Where the broom's handle is under the hand as the wrist camera of @arm shows it from some 10 cm above:
        (base-frame xy on the handle's middle line across from @tip, the handle's top over the floor) or None.
        @direction: the handle's direction in the base frame. Only the handle's part within 6 cm along it of the hand
        is looked at, and only below the fingers."""
        points = camera_points(self.ob.data[WRIST[arm] + 'depth_linear'], self.ob.data['robot_r1::cam_rel_poses'].reshape(3, 7)[SLOT[arm]], FOCAL[arm], stride=2, max_range=.8)
        self.wrist_debug = dict(points=len(points))
        if len(points) < 50:
            return None
        tip = np.asarray(tip, float); z = points[:, 2] + BASE_Z
        along = np.asarray(direction, float)[:2]; along = along / np.linalg.norm(along); across = np.array([-along[1], along[0]])
        d = points[:, :2] - tip[:2]; a = d @ along; b = d @ across
        near = (np.abs(a) < .06) & (np.abs(b) < .25) & (z < min(.15, tip[2] + BASE_Z - .03))
        if near.sum() < 50:
            return None
        # (the floor as this camera sees it stood 1.5-2 cm up, not at 0: the handle is what stands over it)
        floor = float(np.percentile(z[near], 30))
        sel = near & (z > floor + .010)
        self.wrist_debug.update(handle=int(sel.sum()), floor=round(floor, 3))
        if sel.sum() < 15:
            return None
        top = float(np.percentile(z[sel], 95)); band = sel & (z > top - .012)
        width = float(np.percentile(b[band], 95) - np.percentile(b[band], 5))
        self.wrist_debug.update(top=round(top, 3), width=round(width, 3), band=int(band.sum()))
        if width > .05 or band.sum() < 8:
            return None                 # (not one thin handle)
        return tip[:2] + float(np.median(b[band])) * across, top - floor


def observed_pipeline(rt):
    from sweep_pipeline import full_task
    located = yield from localize(rt)
    if not located:
        return 'not_localized'
    return (yield from full_task(rt.ctx, rt.est))
