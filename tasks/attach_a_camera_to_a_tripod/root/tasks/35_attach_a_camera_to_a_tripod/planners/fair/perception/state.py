"""Perception of the fair planner of task 35: the native observation (3 RGB-D cameras + proprio) to a World in the odometry frame.

Input is only the observation dict. The base pose integrates the measured base velocity (odometry frame: the base at the first control).
Kept over time: obstacle hits and floor sightings on a 5 cm grid (routes), the clouds of the last RECENT_UPDATES updates (detection and fits),
the camera and tripod estimates. While the camera is held its pose is predicted from the hand (forward kinematics) and the grasp transform
measured at the grasp, and its points are cut out of the scene. `freeze` stops the refits of an object while an arm works at it.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional

import numpy as np

from core.skills_lib import nav_grid
from core.skills_lib import r1pro_kinematics as K
from core.skills_lib import rgbd_geometry as G

from .. import models as MD
from .. import params as P
from ..world import ObjectEst, World
from . import cloud as C
from . import objects as O


def odom_T(base: np.ndarray) -> np.ndarray:
    c, s = np.cos(base[2]), np.sin(base[2])
    return np.array([[c, -s, 0., base[0]], [s, c, 0., base[1]], [0., 0., 1., 0.], [0., 0., 0., 1.]])


class Perception:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        n = int(2 * P.OBST_HALF_M / P.OBST_RES_M)
        self.obst = np.zeros((n, n), np.uint8)
        self.floor_seen = np.zeros((n, n), bool)
        self.odo = G.Odometry()
        self.floor_z = 0.0
        self.step = 0
        self.recent: deque = deque(maxlen=P.RECENT_UPDATES)
        self.camera: Optional[ObjectEst] = None
        self.tripod: Optional[ObjectEst] = None
        self.held: Optional[Dict[str, Any]] = None       # side, T_mid_cam (camera pose in the finger-midpoint frame)
        self.freeze = {'camera': False, 'tripod': False}
        self.rejected: list = []                          # xy of camera candidates whose fit failed
        self.counts = dict(updates=0, cam_cands=0, cam_fits=0, cam_fail=0, tri_cands=0, tri_fits=0, tri_fail=0)
        self.proprio: Optional[np.ndarray] = None
        self.cam_rel: Optional[np.ndarray] = None
        self.views: list = []
        self.map_hold = False
        self.hint: Dict[str, Any] = {'camera': None, 'tripod': None}

    # ---- frames ------------------------------------------------------------------------------------------------------------------------
    def pose(self) -> np.ndarray:
        return self.odo.pose.copy()

    def mid_T(self, q: np.ndarray, side: str) -> np.ndarray:
        """Finger-midpoint frame (gripper link rotation) of an arm in the odometry frame."""
        servo = K.ArmServo(side)
        servo.use_fingers_mid()
        return odom_T(self.pose()) @ servo.pose(q)

    def held_T(self, q: np.ndarray) -> Optional[np.ndarray]:
        if self.held is None:
            return None
        return self.mid_T(q, self.held['side']) @ self.held['T_mid_cam']

    def grasped(self, q: np.ndarray, side: str, T_cam: np.ndarray) -> None:
        """The hand holds the camera: keep its pose in the finger-midpoint frame (from the estimate at the grasp)."""
        self.held = dict(side=side, T_mid_cam=np.linalg.inv(self.mid_T(q, side)) @ T_cam)
        self.camera = None
        self.hint['camera'] = None
        if P.CLEAR_GRASPED:     # the camera left its place on the floor: forget its obstacle hits
            c = self._cells(T_cam[None, :2, 3])[0]
            r = int(np.ceil(P.CLEAR_GRASPED_R_M / P.OBST_RES_M))
            self.obst[max(c[0] - r, 0):c[0] + r + 1, max(c[1] - r, 0):c[1] + r + 1] = 0

    def released(self) -> None:
        self.held = None
        self.hint['camera'] = None

    def refresh(self, name: str) -> None:
        """Forget an estimate and the recent clouds: the next fit comes from the current pose only. Odometry under-reports turns by a few
        per cent, so an estimate made elsewhere is off in the robot frame; fits made where the arm will act are consistent with it."""
        old = getattr(self, name)
        self.hint[name] = None if old is None or not P.REFRESH_HINT else old.xy   # the refit looks near the estimate it replaces
        setattr(self, name, None)
        self.freeze[name] = False
        self.recent.clear()
        self.counts['refresh'] = self.counts.get('refresh', 0) + 1

    # ---- updates -------------------------------------------------------------------------------------------------------------------------
    def observe(self, obs: Dict[str, Any], step: int, perceive: Optional[bool] = None) -> None:
        proprio = np.asarray(G.as_numpy(obs[G.PROPRIO_KEY]), float).reshape(-1)
        self.proprio = proprio
        self.odo.update(proprio[:3])
        self.step = step
        if perceive is None:
            perceive = step % P.PERCEIVE_EVERY == 0
        if perceive:
            self._perceive(obs, proprio)

    def _perceive(self, obs: Dict[str, Any], proprio: np.ndarray) -> None:
        self.counts['updates'] += 1
        q = K.q_from_proprio(proprio)
        cam_rel = np.asarray(G.as_numpy(obs[G.CAM_POSES_KEY]), float).reshape(-1)
        self.cam_rel = cam_rel
        base = self.pose()
        held = self.held_T(q)
        held_c = None if held is None else (np.linalg.inv(odom_T(base)) @ np.r_[held[:3, :3] @ MD.CAMERA_PTS.mean(0) + held[:3, 3], 1.])[:3]
        cl = C.clouds(obs, q, cam_rel, self.floor_z, held_c)
        self.floor_z = C.floor_height(cl['head'][0], self.floor_z) if self.counts['updates'] > 1 else C.floor_height(cl['head'][0], 0.)
        pts = np.concatenate([G.robot_to_odom(cl[c][0], base) for c in C.CAMS])
        head = G.robot_to_odom(cl['head'][0], base)
        self.recent.append(pts)
        self.views = C.views(obs, cam_rel, base)
        self._layers(pts if P.OBST_FROM_WRISTS else head, head)
        allp = np.concatenate(list(self.recent))
        if self.held is None and not self.freeze['camera']:
            self._camera(allp)
        if not self.freeze['tripod']:
            self._tripod(allp)

    def _cells(self, xy: np.ndarray) -> np.ndarray:
        c = np.floor((xy + P.OBST_HALF_M) / P.OBST_RES_M).astype(int)
        return np.clip(c, 0, self.obst.shape[0] - 1)

    def _layers(self, pts: np.ndarray, head: np.ndarray) -> None:
        if self.map_hold:
            return
        h = pts[:, 2] - self.floor_z
        near = np.hypot(pts[:, 0] - self.pose()[0], pts[:, 1] - self.pose()[1]) < P.OBST_SELF_R_M   # where the robot stands nothing blocks it
        ob = pts[(h > P.OBST_BAND_M[0]) & (h < P.OBST_BAND_M[1]) & ~near]
        if len(ob):
            c = np.unique(self._cells(ob[:, :2]), axis=0)
            self.obst[c[:, 0], c[:, 1]] = np.minimum(self.obst[c[:, 0], c[:, 1]].astype(int) + 1, 250).astype(np.uint8)
        fl = head[np.abs(head[:, 2] - self.floor_z) < P.FLOOR_BAND_M]
        if len(fl):
            c = self._cells(fl[:, :2])
            self.floor_seen[c[:, 0], c[:, 1]] = True

    def _camera(self, pts: np.ndarray) -> None:
        cands = O.camera_candidates(pts, self.floor_z)
        cands = [c for c in cands if all(np.linalg.norm(c['centre'][:2] - r) > 0.12 for r in self.rejected)]
        self.counts['cam_cands'] += len(cands)
        if self.camera is not None:     # refit the known one first (the nearest candidate), otherwise the best supported
            cands.sort(key=lambda c: float(np.linalg.norm(c['centre'][:2] - self.camera.xy)))
            cands = [c for c in cands if np.linalg.norm(c['centre'][:2] - self.camera.xy) < 0.25][:1] or []
        elif self.hint.get('camera') is not None:
            cands = [c for c in cands if np.linalg.norm(c['centre'][:2] - self.hint['camera']) < P.HINT_R_M]
            cands.sort(key=lambda c: float(np.linalg.norm(c['centre'][:2] - self.hint['camera'])))
        else:
            cands.sort(key=lambda c: -c['n'])
        best = None
        free = self.camera is None and self.hint.get('camera') is None
        for cand in cands[:3]:
            est = O.fit_camera(pts, self.floor_z, cand, self.views, self.step)
            if est is None:
                self.counts['cam_fail'] += 1
                if self.camera is None:
                    self.rejected.append(cand['centre'][:2].copy())
                continue
            if not free:
                best = est
                break
            if best is None or est.cost < best.cost:     # no prior: the best fitting candidate, not the one with most points
                best = est
        if best is not None:
            self.counts['cam_fits'] += 1
            self.camera = self._newer(self.camera, best)

    def _newer(self, old: Optional[ObjectEst], est: ObjectEst) -> ObjectEst:
        """The newest fit wins (it agrees with the current odometry pose), unless it rests on far fewer points than a fit made just before."""
        if old is not None and est.n_cloud < P.KEEP_RATIO * old.n_cloud and self.step - old.seen_step < P.KEEP_STEPS:
            old.seen_step = self.step
            return old
        return est

    def _tripod(self, pts: np.ndarray) -> None:
        cands = O.tripod_candidates(pts, self.floor_z)
        self.counts['tri_cands'] += len(cands)
        if self.tripod is not None:
            cands = [c for c in cands if np.linalg.norm(c['centre'][:2] - self.tripod.xy) < 0.4]
        elif self.hint.get('tripod') is not None:
            cands = [c for c in cands if np.linalg.norm(c['centre'][:2] - self.hint['tripod']) < P.HINT_R_M]
        cands.sort(key=lambda c: -c['n_column'])
        for cand in cands[:2]:
            est = O.fit_tripod(pts, self.floor_z, cand, self.views, self.step)
            if est is None:
                self.counts['tri_fail'] += 1
                continue
            self.counts['tri_fits'] += 1
            self.tripod = self._newer(self.tripod, est)
            return

    # ---- products ------------------------------------------------------------------------------------------------------------------------
    def world(self, q: np.ndarray) -> World:
        return World(step=self.step, base=self.pose(), floor_z=self.floor_z, camera=self.camera, tripod=self.tripod, held_T=self.held_T(q))

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
        if self.camera is not None:
            out.update(camera=self.camera.T[:2, 3].round(4).tolist(), camera_yaw=round(float(np.arctan2(self.camera.T[1, 0], self.camera.T[0, 0])), 4),
                       camera_fit=[round(self.camera.cost, 6), round(self.camera.inlier, 3), self.camera.n_cloud, self.camera.seen_step])
        if self.tripod is not None:
            F = self.tripod.T @ MD.TRIPOD_F
            out.update(f=F[:3, 3].round(4).tolist(), f_yaw=round(float(np.arctan2(F[1, 0], F[0, 0])), 4),
                       tripod_fit=[round(self.tripod.cost, 6), round(self.tripod.inlier, 3), self.tripod.n_cloud, self.tripod.seen_step],
                       tripod_yaws=[round(v, 3) for v in self.tripod.yaws], tripod_costs=[round(v, 6) for v in self.tripod.costs])
        held = self.held_T(q)
        if held is not None:
            out['m'] = (held @ MD.CAMERA_M)[:3, 3].round(4).tolist()
        return out
