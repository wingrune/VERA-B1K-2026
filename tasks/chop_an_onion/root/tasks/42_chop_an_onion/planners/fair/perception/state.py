"""Perception of the fair planner of task 42: the native observation (3 RGB-D cameras + proprio) to a World in the odometry frame.

Input is only the observation dict. The base pose integrates the measured base velocity (odometry frame: the base at the first control).
Kept over time: obstacle hits and floor sightings on a 5 cm grid (routes), the clouds of the last RECENT_UPDATES updates, and, when `detect` is on
(the base stands still with the arms at rest: scan and look phases), the counter, the sink basin and the items on it. Heights are z of the robot frame (the floor is about 0). Estimates are replaced by
the newest fit; a role in `frozen` keeps its estimate (the object is in a hand or the arm works at it).
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

from core.skills_lib import nav_grid
from core.skills_lib import r1pro_kinematics as K
from core.skills_lib import rgbd_geometry as G

from .. import params as P
from ..world import BasinEst, CounterEst, ItemEst, World
from . import cloud as C
from . import items as I
from . import scene as S
from . import shapes as SH


def odom_T(base: np.ndarray) -> np.ndarray:
    c, s = np.cos(base[2]), np.sin(base[2])
    return np.array([[c, -s, 0., base[0]], [s, c, 0., base[1]], [0., 0., 1., 0.], [0., 0., 0., 1.]])


class Perception:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        n = int(2 * P.OBST_HALF_M / P.OBST_RES_M)
        self.obst = np.zeros((n, n), np.float32)
        self.floor_seen = np.zeros((n, n), bool)
        self.odo = G.Odometry()
        self.floor_z = 0.0
        self.step = 0
        self.recent: deque = deque(maxlen=P.RECENT_UPDATES)
        self.counter: Optional[CounterEst] = None
        self.ref: Optional[Dict[str, Any]] = None       # the counter frame of the first good view: the landmark that anchors the odometry
        self.basin: Optional[BasinEst] = None
        self.items: Dict[str, ItemEst] = {}
        self.onion: Optional[ItemEst] = None
        self.pieces: Dict[str, Any] = {}
        self.contents: Dict[str, Any] = {}
        self.basin_pts: np.ndarray = np.zeros((0, 3))   # the points seen over the basin and its rim in the last good view
        self.frozen: set = set()
        self.detect = False               # run the scene detection on this update
        self.map_hold = False             # no obstacle updates (the arm moves in the views)
        self.wrist_pts: Dict[str, np.ndarray] = {}      # latest wrist clouds (odometry frame), own body cut out
        self.held_cut: Optional[tuple] = None   # (side, radius): cut the object in this hand out of the clouds
        self.counts = dict(updates=0, counter=0, basin=0, items=0, fits=0, fit_fail=0, anchors=0)
        self.anchor_log: list = []
        self.blacklist: list = []                       # counters looked over without finding the sink (the hob run is at the same height)
        self.last_anchor = -10 ** 9
        self.anchor_yaw_total = 0.0
        self.proprio: Optional[np.ndarray] = None
        self.stall_ref: Optional[tuple] = None          # (step, pose, head depth thumbnail) at the last stall check
        self.stalled_steps = 0                          # controls the base was commanded to move while the head view stood still
        self.cam_rel: Optional[np.ndarray] = None

    # ---- frames ------------------------------------------------------------------------------------------------------------------------
    def pose(self) -> np.ndarray:
        return self.odo.pose.copy()

    def mid_T(self, q: np.ndarray, side: str) -> np.ndarray:
        """Finger-midpoint frame (gripper link rotation) of an arm in the odometry frame."""
        servo = K.ArmServo(side)
        servo.use_fingers_mid()
        return odom_T(self.pose()) @ servo.pose(q)

    def tips_mid(self, q: np.ndarray, side: str) -> np.ndarray:
        """Middle of the two fingertips (odometry frame) from the forward kinematics of both finger links: the real grasp centre, also when the
        fingers are not symmetric (one rests on an object)."""
        poses = K.kinematics().forward(q)[0]
        tips = []
        for k, y in ((1, -0.0163), (2, 0.01575)):                 # fingertips of the two finger links (the grasp points of the native assisted grasp)
            t = poses[f'{side}_gripper_finger_link{k}']
            tips.append(t[:3, 3] + t[:3, :3] @ np.array([0., y, -0.038]))
        mid = np.mean(tips, axis=0)
        return (odom_T(self.pose()) @ np.r_[mid, 1.])[:3]

    # ---- updates -------------------------------------------------------------------------------------------------------------------------
    def observe(self, obs: Dict[str, Any], step: int, perceive: Optional[bool] = None) -> None:
        proprio = np.asarray(G.as_numpy(obs[G.PROPRIO_KEY]), float).reshape(-1)
        self.proprio = proprio
        self.odo.update(proprio[:3])
        self.step = step
        self._stall_check(obs, step)
        if perceive is None:
            perceive = step % P.PERCEIVE_EVERY == 0
        if perceive:
            self._perceive(obs, proprio)

    def _stall_check(self, obs: Dict[str, Any], step: int) -> None:
        """The measured base velocity is the commanded one when the base is jammed against furniture: if the integrated pose moved but the head depth
        did not change, undo the move (the base is stuck, the heading must not drift)."""
        if step % P.STALL_EVERY != 0:
            return
        depth = np.squeeze(np.asarray(G.as_numpy(obs[G.depth_key('head')]), float))
        thumb = depth[::P.STALL_STRIDE, ::P.STALL_STRIDE]
        pose = self.odo.pose
        ref, self.stall_ref = self.stall_ref, (step, pose.copy(), thumb)
        if ref is None or ref[2].shape != thumb.shape:
            return
        moved = float(np.hypot(*(pose[:2] - ref[1][:2]))) > P.STALL_MIN_MOVE_M or abs(G.wrap(pose[2] - ref[1][2])) > P.STALL_MIN_TURN_RAD
        ok = np.isfinite(thumb) & np.isfinite(ref[2]) & (thumb > 0.1) & (ref[2] > 0.1) & (thumb < P.STALL_MAX_DEPTH_M) & (ref[2] < P.STALL_MAX_DEPTH_M)
        if not moved or ok.sum() < 20:
            self.stalled_steps = 0
            return
        if float(np.mean(np.abs(thumb[ok] - ref[2][ok]))) < P.STALL_DEPTH_M:
            pose[:] = ref[1]                           # jammed: undo the integrated move of this window
            self.stall_ref = (step, pose.copy(), thumb)
            self.stalled_steps += P.STALL_EVERY
            self.counts['stalled'] = self.counts.get('stalled', 0) + 1
        else:
            self.stalled_steps = 0

    def _perceive(self, obs: Dict[str, Any], proprio: np.ndarray) -> None:
        self.counts['updates'] += 1
        q = K.q_from_proprio(proprio)
        cam_rel = np.asarray(G.as_numpy(obs[G.CAM_POSES_KEY]), float).reshape(-1)
        self.cam_rel = cam_rel
        base = self.pose()
        held = None
        if self.held_cut is not None:
            side, radius = self.held_cut
            servo = K.ArmServo(side)
            servo.use_fingers_mid()
            held = (servo.pose(q)[:3, 3], radius)
        cl = C.clouds(obs, q, cam_rel, self.floor_z, held)
        self.floor_z = P.FLOOR_Z_M                     # the base stands on the floor: the robot frame's floor is fixed (a cloud estimate drifted to -0.1 m and turned the floor into obstacles)
        pts = np.concatenate([G.robot_to_odom(cl[c][0], base) for c in C.CAMS])
        head = G.robot_to_odom(cl['head'][0], base)
        self.wrist_pts = {'left': G.robot_to_odom(cl['left_wrist'][0], base), 'right': G.robot_to_odom(cl['right_wrist'][0], base)}
        if not self.map_hold and abs(float(proprio[2])) < P.LAYERS_MAX_TURN_RATE:      # no obstacle hits while turning: the heading integrated in a turn is off by a few per cent
            self._layers(head)
        if self.detect:
            moving = bool(np.abs(proprio[:3]).max() > P.MOVING_SPEED)
            if moving:                      # clouds taken while the base moves do not line up: this frame only
                self.recent.clear()
            self.recent.append(pts)
            self._scene(np.concatenate(list(self.recent)), moving)
        else:
            self.recent.clear()

    def _cells(self, xy: np.ndarray) -> np.ndarray:
        c = np.floor((xy + P.OBST_HALF_M) / P.OBST_RES_M).astype(int)
        return np.clip(c, 0, self.obst.shape[0] - 1)

    def _layers(self, head: np.ndarray) -> None:
        h = head[:, 2] - self.floor_z
        near = np.hypot(head[:, 0] - self.pose()[0], head[:, 1] - self.pose()[1]) < P.OBST_SELF_R_M
        ob = head[(h > P.OBST_BAND_M[0]) & (h < P.OBST_BAND_M[1]) & ~near]
        self.obst *= P.OBST_DECAY                      # what is not seen again fades: the odometry corrections smear old hits
        if len(ob):
            c = np.unique(self._cells(ob[:, :2]), axis=0)
            self.obst[c[:, 0], c[:, 1]] = np.minimum(self.obst[c[:, 0], c[:, 1]] + 1.5, P.OBST_MAX_HITS)
        fl = head[np.abs(head[:, 2] - self.floor_z) < P.FLOOR_BAND_M]
        if len(fl):
            c = self._cells(fl[:, :2])
            self.floor_seen[c[:, 0], c[:, 1]] = True

    def _anchor(self, ce: CounterEst, ba: Optional[BasinEst]) -> bool:
        """Pull the odometry pose onto the counter frame of the first good view (the counter is the landmark; its orientation, its front edge and
        the basin on it are seen again and again): heading first, then the distance to the front edge, then the position along the counter from the
        basin. The wheel odometry alone drifts by tenths of a radian over a few metres. Returns True when the pose was changed."""
        if ce.length < P.ANCHOR_MIN_LENGTH_M or P.ANCHOR_GAIN <= 0.0:
            return False
        th = math.atan2(ce.u[1], ce.u[0])
        front = float(ce.n @ ce.centre) + 0.5 * ce.depth
        if self.ref is None:
            self.ref = dict(th=th, n=ce.n.copy(), u=ce.u.copy(), front=front, basin=None)
            return False
        ref = self.ref
        if self.ref['basin'] is None and ba is not None:
            self.ref['basin'] = ba.centre.copy()
        pose = self.odo.pose
        dth = (ref['th'] - th + math.pi / 2) % math.pi - math.pi / 2
        if abs(dth) > P.ANCHOR_YAW_DEAD_RAD and abs(dth) < P.ANCHOR_MAX_YAW_RAD and abs(self.anchor_yaw_total) < P.ANCHOR_TOTAL_YAW_RAD:
            pose[2] += P.ANCHOR_GAIN * dth
            self.anchor_yaw_total += P.ANCHOR_GAIN * dth
            self.counts['anchors'] += 1
            self.anchor_log.append((self.step, 'yaw', round(dth, 4)))
            self.last_anchor = self.step
            self._decay_grid(abs(dth) > P.ANCHOR_CLEAR_YAW_RAD)
            return True
        n = ref['n'] if float(ce.n @ ref['n']) > 0 else -ref['n']
        s_n = float(n @ ce.centre) + 0.5 * ce.depth - front
        if abs(s_n) > P.ANCHOR_POS_DEAD_M and abs(s_n) < P.ANCHOR_MAX_POS_M:
            pose[:2] -= P.ANCHOR_GAIN * s_n * n
            self.counts['anchors'] += 1
            self.anchor_log.append((self.step, 'normal', round(s_n, 4)))
            self.last_anchor = self.step
            self._decay_grid(abs(s_n) > P.ANCHOR_CLEAR_POS_M)
            return True
        if ba is not None and ref['basin'] is not None:
            s_u = float(ref['u'] @ (ba.centre - ref['basin']))
            if abs(s_u) > P.ANCHOR_POS_DEAD_M and abs(s_u) < P.ANCHOR_MAX_POS_M:
                pose[:2] -= P.ANCHOR_GAIN * s_u * ref['u']
                self.counts['anchors'] += 1
                self.anchor_log.append((self.step, 'along', round(s_u, 4)))
                self.last_anchor = self.step
                self._decay_grid(abs(s_u) > P.ANCHOR_CLEAR_POS_M)
                return True
        return False

    def _decay_grid(self, big: bool) -> None:
        """After a pose correction the obstacle hits of the old frame are off by as much as the correction (more at range): forget them, the head
        sees the obstacles again at once."""
        if big:
            self.obst[:] = 0.0

    def _scene(self, pts: np.ndarray, moving: bool = False) -> None:
        res = S.find_counter(pts, self.pose()[:2], self.step, self.blacklist)
        if res is None:
            return
        ce = res[0]
        ba_now = S.find_basin(pts, ce, self.step)
        if self._anchor(ce, ba_now):
            self.recent.clear()                          # its clouds were registered with the old pose
            return                                       # the base pose was corrected: this view is in the old frame, the next one is not
        if self.counter is not None and self.step - self.counter.seen_step < P.COUNTER_LOCK_STEPS:
            dth = abs((math.atan2(ce.u[1], ce.u[0]) - math.atan2(self.counter.u[1], self.counter.u[0]) + math.pi / 2) % math.pi - math.pi / 2)
            if dth > P.COUNTER_LOCK_RAD:                 # another run of counters (the kitchen is L-shaped): the one first seen is kept unless this one has the sink
                if ba_now is None or self.basin is not None:
                    self.counts['counter_rejected'] = self.counts.get('counter_rejected', 0) + 1
                    return
                self.basin, self.items, self.onion = None, {}, None
                self.counter = None
        if self.counter is None or ce.length >= 0.7 * self.counter.length or self.step - self.counter.seen_step > 90:
            if self.counter is not None:                # keep the long side's orientation sign of the first fit (the front stays the front)
                ce.n = ce.n if float(ce.n @ self.counter.n) > 0 else -ce.n
            self.counter = ce
        self.counts['counter'] += 1
        ce = self.counter
        ba = ba_now if ce is res[0] and ba_now is not None else S.find_basin(pts, ce, self.step)
        if ba is not None:
            self.basin = ba
            self.counts['basin'] += 1
        cand = I.find_items(pts, ce, self.basin, self.step)
        for role, lst in cand.items():
            if role in self.frozen or not lst or (moving and role in self.items):
                continue
            best = lst[0]
            if role in ('board', 'parer'):
                fit = SH.fit_shape(best, pts, ce, self.step)
                self.counts['fits' if fit is not None else 'fit_fail'] += 1
                if fit is None:
                    if role not in self.items or self.items[role].fit == 0.0:
                        self.items[role] = best
                    continue
                best = fit
            self.items[role] = best
            self.counts['items'] += 1
        if self.basin is not None and 'onion' not in self.frozen:
            on = I.find_onion(pts, ce, self.basin, self.step)
            if on is not None:
                self.onion = on
        if 'bowl' in self.items:
            self.pieces = I.find_pieces(pts, ce, self.items['bowl'], self.step)
        if self.basin is not None:
            self.contents = I.basin_contents(pts, ce, self.basin)
            near = pts[np.linalg.norm(pts[:, :2] - self.basin.centre, axis=1) < P.BASIN_PTS_R_M]
            if len(near) > 200:
                self.basin_pts = near

    # ---- products ------------------------------------------------------------------------------------------------------------------------
    def world(self, q: np.ndarray) -> World:
        items = dict(self.items)
        if self.onion is not None:
            items['onion'] = self.onion
        return World(step=self.step, base=self.pose(), floor_z=self.floor_z, counter=self.counter, basin=self.basin, items=items)

    def refine_onion(self, side: str, item: ItemEst) -> Optional[ItemEst]:
        """The onion again from the wrist camera of the hand that is about to take it (above the basin the rim no longer hides it): the same blob
        search as the head's, on the wrist cloud near the estimate (the basin walls are excluded by the shrunk basin box)."""
        pts = self.wrist_pts.get(side)
        if pts is None or self.basin is None or self.counter is None:
            return None
        pts = pts[np.linalg.norm(pts[:, :2] - item.centre[:2], axis=1) < P.REFINE_R_M]
        if len(pts) < P.REFINE_MIN_POINTS:
            return None
        return I.find_onion(pts, self.counter, self.basin, self.step)

    def reject_counter(self) -> None:
        """The counter in view has no sink basin: remember its rectangle and start over (scan for another run of counters)."""
        ce = self.counter
        if ce is not None:
            self.blacklist.append(dict(centre=ce.centre.copy(), u=ce.u.copy(), n=ce.n.copy(), length=ce.length + 0.4, depth=ce.depth))
        self.counter, self.basin, self.items, self.onion, self.pieces = None, None, {}, None, {}
        self.recent.clear()
        self.obst[:] = 0.0                              # the hits of the walk so far were registered with a drifting odometry

    def forget(self, role: str) -> None:
        if role == 'onion':
            self.onion = None
        self.items.pop(role, None)
        self.recent.clear()

    def nav_grid(self, lo: np.ndarray, hi: np.ndarray) -> nav_grid.Grid:
        """Obstacle grid (GRID_RES) over [lo, hi] from the obstacle hits; cells never seen as floor cost UNKNOWN_COST more."""
        grid = nav_grid.Grid(lo, hi, P.GRID_RES)
        ix, iy = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]), indexing='ij')
        centres = np.stack([lo[0] + (ix + .5) * P.GRID_RES, lo[1] + (iy + .5) * P.GRID_RES], -1)
        cc = self._cells(centres.reshape(-1, 2)).reshape(centres.shape)
        grid.blocked = self.obst[cc[..., 0], cc[..., 1]] >= P.OBST_MIN_HITS
        near = np.linalg.norm(centres - self.pose()[:2], axis=-1) < 0.8
        seen = self.floor_seen[cc[..., 0], cc[..., 1]] | near
        grid.extra_cost = np.where(seen, 0., P.UNKNOWN_COST)
        return grid

    def est(self, q: np.ndarray) -> Dict[str, Any]:
        """Estimates in the odometry frame for the harness log (the harness compares them to the truth; never fed back)."""
        base = self.pose()
        out: Dict[str, Any] = dict(base=base[:2].round(4).tolist(), base_yaw=round(float(base[2]), 4), floor_z=round(self.floor_z, 4),
                                   counts=dict(self.counts))
        if self.counter is not None:
            c = self.counter
            out['counter'] = c.centre.round(4).tolist()
            out['counter_top'] = round(c.top, 4)
            out['counter_n'] = c.n.round(3).tolist()
            out['counter_size'] = [round(c.length, 3), round(c.depth, 3)]
        if self.basin is not None:
            out['basin'] = self.basin.centre.round(4).tolist()
            out['basin_floor'] = round(self.basin.floor, 4)
        for role, it in self.items.items():
            out[role] = it.centre.round(4).tolist()
            if it.yaw is not None:
                out[role + '_yaw'] = round(float(it.yaw), 3)
            out[role + '_fit'] = [round(it.fit, 3), it.n_points, it.seen_step]
        if self.onion is not None:
            out['onion'] = self.onion.centre.round(4).tolist()
            out['onion_size'] = self.onion.size.round(3).tolist()
        if self.contents:
            out['basin_contents'] = self.contents
        if self.pieces.get('n'):
            out['pieces'] = dict(n=self.pieces['n'], top=round(self.pieces['top'], 3), blobs=len(self.pieces['blobs']))
        return out
