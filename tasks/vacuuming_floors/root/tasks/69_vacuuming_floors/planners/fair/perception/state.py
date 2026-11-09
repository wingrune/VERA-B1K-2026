"""Perception of the fair planner of task 69: the native observation (3 RGB-D cameras + proprio) to a World in the odometry frame.

Input is only the observation dict. The base pose integrates the measured base velocity (odometry frame: the base at the first control).
Kept over time: obstacle hits and floor sightings on a 5 cm grid (routes), the clouds of the last RECENT_UPDATES updates (vacuum detection
and fits), the vacuum estimate, the dust memory (dark flakes on the floor), the marker colour. While the vacuum is held its pose is predicted
from the hand (forward kinematics) and the grasp transform measured at the grasp (corrected by the marker position seen by the head camera),
and its points are cut out of the scene. `freeze` stops the refits of the vacuum while an arm works at it.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

from core.skills_lib import r1pro_kinematics as K
from core.skills_lib import rgbd_geometry as G

from .. import models as MD
from .. import params as P
from ..world import Dust, VacuumEst, World
from . import cloud as C
from .gridmap import MapMixin
from .cloud import odom_T
from . import localize as L
from . import objects as O
from . import track as T


class Perception(MapMixin):
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        n = int(2 * P.OBST_HALF_M / P.OBST_RES_M)
        self.obst = np.zeros((n, n), np.uint8)
        self.obst_layers = np.zeros((n, n), np.uint8)      # bit k: a hit in height layer k (low, middle, high)
        self.floor_seen = np.zeros((n, n), bool)
        self.dust_cov = np.zeros((n, n), bool)             # floor looked at for flakes: 0.9-2.6 m from the head while the base stood
        self.odo = G.Odometry()
        self.floor_z = 0.0
        self.floor_known = False
        self.still_steps = 0
        self.loc_step = -10 ** 6
        self.was_on = False
        self.anchor = None
        self.loc_log: list = []
        self.yaw_lock = False
        self.tilt_alarm = -10 ** 6
        self.marker_lost = 0
        self.step = 0
        self.recent: deque = deque(maxlen=P.RECENT_UPDATES)
        self.vacuum: Optional[VacuumEst] = None
        self.held: Optional[Dict[str, Any]] = None       # side, T_mid_vac (vacuum pose in the finger-midpoint frame)
        self.freeze = {'vacuum': False}
        self.rejected: list = []                          # xy of vacuum candidates whose fit failed
        self.dust: List[Dust] = []
        self.next_pid = 1
        self.marker: Dict[str, Any] = dict(state=None, step=-1, votes=0, pending=None, seen=0)
        self.counts = dict(updates=0, vac_cands=0, vac_fits=0, vac_fail=0, dust_seen=0, dust_new=0, dust_gone=0, marker_fix=0)
        self.proprio: Optional[np.ndarray] = None
        self.cam_rel: Optional[np.ndarray] = None
        self.map_hold = False
        self.hint: Dict[str, Any] = {'vacuum': None}
        self.dust_on = True
        self.marker_fix = True

    # ---- frames ------------------------------------------------------------------------------------------------------------------------
    def pose(self) -> np.ndarray:
        return self.odo.pose.copy()

    def mid_T(self, q: np.ndarray, side: str) -> np.ndarray:
        """Finger-midpoint frame (gripper link rotation) of an arm...

        Finger-midpoint frame (gripper link rotation) of an arm in the odometry frame."""
        servo = K.ArmServo(side)
        servo.use_fingers_mid()
        return odom_T(self.pose()) @ servo.pose(q)

    def held_T(self, q: np.ndarray) -> Optional[np.ndarray]:
        if self.held is None:
            return None
        return self.mid_T(q, self.held['side']) @ self.held['T_mid_vac']

    def grasped(self, q: np.ndarray, side: str, T_vac: np.ndarray) -> None:
        """The hand holds the vacuum.

        The hand holds the vacuum: keep its pose in the finger-midpoint frame (from the estimate at the grasp)."""
        self.held = dict(side=side, T_mid_vac=np.linalg.inv(self.mid_T(q, side)) @ T_vac)
        self.vacuum = None
        self.hint['vacuum'] = None
        self.marker = dict(state='on' if self.was_on else None, step=-1, votes=0, pending=None, seen=0)      # a vacuum put down while on is still on
        self.was_on = False
        c = self._cells(T_vac[None, :2, 3])[0]
        r = int(np.ceil(0.30 / P.OBST_RES_M))      # the vacuum left its place: forget the obstacle hits where it stood
        self.obst[max(c[0] - r, 0):c[0] + r + 1, max(c[1] - r, 0):c[1] + r + 1] = 0
        self.obst_layers[max(c[0] - r, 0):c[0] + r + 1, max(c[1] - r, 0):c[1] + r + 1] = 0

    def released(self) -> None:
        self.held = None
        self.hint['vacuum'] = None

    def refresh(self, name: str) -> None:
        """Forget the vacuum estimate and the recent clouds.

        Forget the vacuum estimate and the recent clouds: the next fit comes from the current pose only (near the old estimate)."""
        old = getattr(self, name)
        self.hint[name] = None if old is None else dict(xy=old.xy, yaw=float(np.arctan2(old.T[1, 0], old.T[0, 0])))
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
        moving = float(np.hypot(proprio[0], proprio[1])) > P.STILL_V or abs(float(proprio[2])) > P.STILL_W
        self.still_steps = 0 if moving else self.still_steps + 1
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
        held_robot = None if held is None else np.linalg.inv(odom_T(base)) @ held
        cl = C.clouds(obs, q, cam_rel, self.floor_z, held_robot)
        self.floor_z = C.floor_height(cl['head'][0], self.floor_z, first=not self.floor_known)
        self.floor_known = self.floor_known or self.floor_z != 0.
        if P.LOC_ON and self.anchor is None and self.step >= P.LOC_ANCHOR_STEP:
            self.anchor = self.blocked_cells().copy()      # the map after the first turn, laid down without drift
        if P.LOC_ON and self.still_steps >= P.STILL_STEPS and self.step - self.loc_step >= P.LOC_EVERY:
            self.loc_step = self.step
            L.correct(self, cl['head'][0])
            base = self.pose()
        pts = np.concatenate([G.robot_to_odom(cl[c][0], base) for c in C.CAMS])
        head = G.robot_to_odom(cl['head'][0], base)
        self.recent.append(pts)
        self._layers(pts if P.OBST_FROM_WRISTS else head, head)
        if self.still_steps >= P.STILL_STEPS:
            self._covered(head, base)
        if self.held is None and not self.freeze['vacuum']:
            self._vacuum(np.concatenate(list(self.recent)))
        vac = held if held is not None else (None if self.vacuum is None else self.vacuum.T)
        rgbd = {c: (G.as_numpy(obs[G.rgb_key(c)]), G.as_numpy(obs[G.depth_key(c)])) for c in C.CAMS}
        if self.dust_on and self.still_steps >= P.STILL_STEPS:     # the odometry lags the motion: flakes are placed only while the base stands
            T.dust(self, rgbd['head'], cam_rel, q, base, held_robot)
        if vac is not None:
            T.marker(self, rgbd, cam_rel, base, vac, q)

    # ---- the vacuum ----------------------------------------------------------------------------------------------------------------------
    def _vacuum(self, pts: np.ndarray) -> None:
        if self.vacuum is not None:                          # refit the known one first
            known = dict(xy=self.vacuum.xy, yaw=float(np.arctan2(self.vacuum.T[1, 0], self.vacuum.T[0, 0])))
            est = O.fit_vacuum(pts, self.floor_z, self.vacuum.xy, self.step, roi=0.25, hint=known, yaw_deg=P.YAW_LOCK_DEG if self.yaw_lock else None)
            if est is not None:
                self.counts['vac_fits'] += 1
                self.vacuum = self._newer(self.vacuum, est)
            return
        cands = O.vacuum_candidates(pts, self.floor_z)
        cands = [c for c in cands if all(np.linalg.norm(c['centre'][:2] - r) > 0.15 for r in self.rejected)]
        self.counts['vac_cands'] += len(cands)
        hint = self.hint.get('vacuum')
        if hint is not None:
            cands = [c for c in cands if np.linalg.norm(c['centre'][:2] - hint['xy']) < 0.5]
            cands.sort(key=lambda c: float(np.linalg.norm(c['centre'][:2] - hint['xy'])))
        else:
            cands.sort(key=lambda c: -c['n'])
        for cand in cands[:2]:
            est = O.fit_vacuum(pts, self.floor_z, cand['centre'][:2], self.step, hint=hint)
            if est is None:
                self.counts['vac_fail'] += 1
                self.rejected.append(cand['centre'][:2].copy())
                continue
            self.counts['vac_fits'] += 1
            self.vacuum = est
            return

    def _newer(self, old: VacuumEst, est: VacuumEst) -> VacuumEst:
        """The newest fit wins (it agrees with the current odometry pose).

        The newest fit wins (it agrees with the current odometry pose), unless it rests on far fewer points than the last one."""
        if est.n_cloud < 0.3 * old.n_cloud and self.step - old.seen_step < 30:
            old.seen_step = self.step
            return old
        return est

    # ---- the marker ------------------------------------------------------------------------------------------------------------------------
    # ---- the dust ------------------------------------------------------------------------------------------------------------------------
    # ---- products ------------------------------------------------------------------------------------------------------------------------
    def toggled(self) -> Optional[bool]:
        s = self.marker.get('state')
        return None if s is None else s == 'on'

    def world(self, q: np.ndarray) -> World:
        live = [d for d in self.dust if not d.gone and d.n >= P.DUST_CONFIRM]
        return World(step=self.step, base=self.pose(), floor_z=self.floor_z, vacuum=self.vacuum, held_T=self.held_T(q), dust=live,
                     toggled=self.toggled(), tilted=self.step - self.tilt_alarm < P.TILT_ALARM_STEPS)

    def est(self, q: np.ndarray) -> Dict[str, Any]:
        """Estimates in the odometry frame for the harness log.

        Estimates in the odometry frame for the harness log (the harness compares them to the truth; never fed back)."""
        base = self.pose()
        out: Dict[str, Any] = dict(base=base[:2].round(4).tolist(), base_yaw=round(float(base[2]), 4), floor_z=round(self.floor_z, 4),
                                   counts=dict(self.counts), marker=dict(state=self.marker.get('state'), last=self.marker.get('last'),
                                                                         fix=self.marker.get('fix'), tilt=self.marker.get('tilt_offset'), lost=self.marker_lost))
        vac = self.held_T(q)
        if vac is None and self.vacuum is not None:
            vac = self.vacuum.T
            out['vac_fit'] = [round(self.vacuum.cost, 6), round(self.vacuum.inlier, 3), self.vacuum.n_cloud, self.vacuum.seen_step]
        if vac is not None:
            out.update(vac=vac[:2, 3].round(4).tolist(), vac_yaw=round(float(np.arctan2(vac[1, 0], vac[0, 0])), 4),
                       cube=(vac[:3, :3] @ MD.CUBE_CENTRE + vac[:3, 3]).round(4).tolist(), held=self.held is not None)
        cells = np.argwhere(self.blocked_cells())
        if len(cells):
            xy = np.unique(np.round((cells * P.OBST_RES_M - P.OBST_HALF_M) / 0.1) * 0.1, axis=0)
            out['map'] = xy[:2500].round(1).tolist()                                 # the blocked cells at 10 cm (debug: compared with the scene by the harness)
        out['dust'] = [d.xy.round(3).tolist() for d in self.dust if not d.gone and d.n >= P.DUST_CONFIRM]
        out['dust_all'] = len(self.dust)
        out['loc'] = self.loc_log[-12:]
        return out
