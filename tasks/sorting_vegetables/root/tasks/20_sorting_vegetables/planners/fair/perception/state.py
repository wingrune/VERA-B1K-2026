"""Perception of the fair planner of task 20: the native observation (3 RGB-D cameras + proprio, camera calibration) to a World in the
odometry frame.

Input is only the observation dict. The base pose integrates the measured base velocity (odometry frame: the base at the first control).
Kept over time: the voxel occupancy of the head views (routes and the arm's body checks), the bowl, basket and vegetable tracks. Detection
runs while the base stands (nearly) still, on the views of the head and, in the look phases, of a wrist. Heights are the robot frame's z
(the floor is 0)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.skills_lib import r1pro_kinematics as K
from core.skills_lib import rgbd_geometry as G

from .. import params as P
from ..geometry import rot_yaw
from ..world import BasketEst, BowlEst, VegEst, World
from . import bowls as B
from . import cloud as C
from . import vegs as V
from .voxmap import VoxMap


def odom_T(base: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:2, 3] = rot_yaw(float(base[2])), base[:2]
    return T


class Perception:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.odo = G.Odometry()
        self.vox = VoxMap()
        self.step = 0
        self.bowls: List[BowlEst] = []
        self.baskets: List[BasketEst] = []
        self.vegs: List[VegEst] = []
        self.detect = False              # run the detection on this update (the planner sets it by phase)
        self.map_hold = False            # no occupancy update (an arm moves through the head view)
        self.wrist: Optional[str] = None  # also detect vegetables in this wrist camera's view
        self.expire_ok = False           # the arms are tucked: a vegetable in the head's view that is not found is gone (the planner sets it)
        self.expire_step = -10 ** 6
        self.held_cut: Optional[Tuple[str, float]] = None   # (side, radius): cut the object in this hand out of the clouds
        self.counts: Dict[str, int] = dict(updates=0, detections=0, bowl_views=0, veg_views=0)
        self.proprio: Optional[np.ndarray] = None
        self.depth_prev: Optional[np.ndarray] = None
        self.loc_step = -10 ** 6
        self.loc_total = np.zeros(3)     # sum of the corrections applied (dx, dy, dyaw), for the trace
        self.psi0: Optional[float] = None    # direction of the house's axes in the odometry frame (mod 90 degrees)
        self.psi_votes: List[float] = []
        self.big_corr: Optional[float] = None     # a large heading correction waiting for a second view to agree
        self.last_loc: Dict[str, Any] = {}
        self.last: Dict[str, Any] = {}
        self.low_cloud: Optional[np.ndarray] = None

    def pose(self) -> np.ndarray:
        return self.odo.pose.copy()

    def mid_T(self, q: np.ndarray, side: str) -> np.ndarray:
        """Finger-midpoint frame (gripper link rotation) of an arm in the odometry frame."""
        servo = K.ArmServo(side)
        servo.use_fingers_mid()
        return odom_T(self.pose()) @ servo.pose(q)

    # ---- updates -------------------------------------------------------------------------------------------------------------------
    def observe(self, obs: Dict[str, Any], step: int, perceive: Optional[bool] = None) -> None:
        proprio = np.asarray(G.as_numpy(obs[G.PROPRIO_KEY]), float).reshape(-1)
        self.proprio = proprio
        vel = np.r_[proprio[:2], P.ODO_TURN_SCALE * proprio[2]]
        if self.still(obs) and np.abs(vel).max() > P.STILL_VEL:     # pressed on an obstacle the base reports a turn it does not make
            vel[:] = 0.                                               # (odometry spun 0.6 rad while the base stood, native t20f16)
            self.counts['still'] = self.counts.get('still', 0) + 1
        self.odo.update(vel)
        self.step = step
        if perceive is None:
            perceive = step % P.PERCEIVE_EVERY == 0
        if perceive:
            self._perceive(obs, proprio)

    def still(self, obs: Dict[str, Any]) -> bool:
        """The head's view did not change since the last control: a coarse depth grid of its upper two thirds (the arms are below) moved
        less than STILL_DEPTH_M on average."""
        d = G.as_numpy(obs[G.depth_key('head')])
        d = np.asarray(d, float).reshape(d.shape[-2], d.shape[-1])[: 2 * d.shape[-2] // 3:P.STILL_STRIDE, ::P.STILL_STRIDE]
        d = np.where(np.isfinite(d) & (d > 0.05) & (d < 8.), d, np.nan)
        prev, self.depth_prev = self.depth_prev, d
        if prev is None or prev.shape != d.shape:
            return False
        diff = np.abs(d - prev)
        ok = np.isfinite(diff)
        return bool(ok.mean() > 0.3 and float(np.nanmean(diff)) < P.STILL_DEPTH_M)

    def _perceive(self, obs: Dict[str, Any], proprio: np.ndarray) -> None:
        self.counts['updates'] += 1
        q = K.q_from_proprio(proprio)
        cam_rel = np.asarray(G.as_numpy(obs[G.CAM_POSES_KEY]), float).reshape(-1)
        base = self.pose()
        held = None
        if self.held_cut is not None:
            side, radius = self.held_cut
            servo = K.ArmServo(side)
            servo.use_fingers_mid()
            held = (servo.pose(q)[:3, 3], radius)
        pts, _, col = C.camera(obs, 'head', q, cam_rel, P.HEAD_STRIDE, held, P.VIEW_RANGE_M)
        moving = bool(np.abs(proprio[:2]).max() > P.MOVING_SPEED or abs(float(proprio[2])) > P.MOVING_SPEED)
        if abs(float(proprio[2])) < P.LOC_MAX_TURN and self.step - self.loc_step >= P.LOC_EVERY:
            self.heading(pts)
            base = self.pose()
        pts = G.robot_to_odom(pts, base)
        if not self.map_hold and abs(float(proprio[2])) < P.TURN_RATE_MAX_MAP:
            cam_o = G.robot_to_odom(G.cam_pose(cam_rel, 'head')[:3, 3][None], base)[0]
            self.vox.add(pts, origin=cam_o)
            r = P.FOOTPRINT_CLEAR_M                       # phantom obstacles (a painted held vegetable, a pose jump) under the robot go
            self.vox.clear_box(np.r_[base[:2] - r, 0.], np.r_[base[:2] + r, P.NAV_BAND_M[1]])
        if not self.detect or moving:
            return
        self.counts['detections'] += 1
        seen = B.find_bowls(pts, col, self.step)
        if seen:
            self.counts['bowl_views'] += 1
            shift = self.bowl_shift(seen)
            jump = shift is None
            if jump:
                shift = self.bowl_jump(seen)
            if shift is not None:        # the base drifted: move it so the bowls seen sit on their tracks, and the view with it
                self.odo.pose[:2] -= shift
                self.loc_total[:2] -= shift
                self.counts['loc_xy'] = self.counts.get('loc_xy', 0) + 1
                base = self.pose()
                pts = pts - np.r_[shift, 0.]
                for b in seen:
                    b.centre = b.centre - np.r_[shift, 0.]
                if jump:                 # a large jump: the occupancy built from the drifted poses is displaced, rebuild it from this view
                    self.counts['loc_jump'] = self.counts.get('loc_jump', 0) + 1
                    self.vox = VoxMap()
                    self.vox.add(pts)
            B.merge(self.bowls, seen)
        views = [(pts, col)]
        if self.wrist is not None:
            wp, _, wc = C.camera(obs, f'{self.wrist}_wrist', q, cam_rel, P.WRIST_STRIDE, held, 1.5)
            views.append((G.robot_to_odom(wp, base), wc))
        low = [vp[vp[:, 2] < P.LOW_CLOUD_TOP_M] for vp, _ in views]
        self.low_cloud = np.concatenate(low) if low else None
        for vp, vc in views:
            found = V.find_vegs(vp, vc, self.step)
            if found:
                self.counts['veg_views'] += 1
                V.merge(self.vegs, found, self.step)
            V.update_baskets(self.baskets, vp, vc, self.step)
        if self.expire_ok and self.step - self.expire_step >= P.VEG_EXPIRE_EVERY:
            self.expire_step = self.step
            self.expire(G.cam_pose(cam_rel, 'head'), base)
        V.label_groups(self.vegs, self.baskets)

    def expire(self, cam_T: np.ndarray, base: np.ndarray) -> None:
        """A vegetable track whose spot is well inside this standing head view (within VEG_EXPIRE_RANGE_M) and that the view did not find
        gets a miss; VEG_MISSES misses drop it (a vegetable moved by the robot, or a sighting from a drifted pose)."""
        keep = []
        inv = np.linalg.inv(cam_T)
        for v in self.vegs:
            if v.seen_step == self.step:
                v.misses = 0
                keep.append(v)
                continue
            p = G.odom_to_robot(v.centre[None], base)[0]
            c = inv[:3, :3] @ p + inv[:3, 3]                 # camera axes (OpenGL): the view looks along -z
            depth = -c[2]
            inside = depth > 0.2 and max(abs(c[0]), abs(c[1])) < P.VEG_EXPIRE_TAN * depth and np.linalg.norm(p[:2]) < P.VEG_EXPIRE_RANGE_M
            v.misses += int(inside)
            if v.misses < P.VEG_MISSES:
                keep.append(v)
            else:
                self.counts['expired'] = self.counts.get('expired', 0) + 1
        self.vegs = keep

    def bowl_shift(self, seen: List[BowlEst]) -> Optional[np.ndarray]:
        """Half the mean offset of bowl sightings from their well-seen tracks, when at least two bowls in this view agree (a bowl pushed by
        the robot moves alone; the base position drifted 0.2-0.4 m over drives with bumps, t20f15). None otherwise."""
        offs = []
        for b in seen:
            d = [float(np.linalg.norm(t.centre[:2] - b.centre[:2])) for t in self.bowls if t.n >= P.RELOC_MIN_SIGHTINGS]
            if d and min(d) < P.RELOC_MAX_M:
                t = [t for t in self.bowls if t.n >= P.RELOC_MIN_SIGHTINGS][int(np.argmin(d))]
                offs.append(b.centre[:2] - t.centre[:2])
        if len(offs) < 2:
            return None
        offs = np.array(offs)
        mean = offs.mean(0)
        if np.max(np.linalg.norm(offs - mean, axis=1)) > P.RELOC_AGREE_M or np.linalg.norm(mean) < P.RELOC_MIN_M:
            return None
        return P.RELOC_GAIN * mean

    def bowl_jump(self, seen: List[BowlEst]) -> Optional[np.ndarray]:
        """Large drift: the bowls seen (two or more) match no track where they are, but one translation puts at least two of them on
        distinct well-seen tracks within 0.2 m, and no other translation (0.3 m apart) does as well. The yaw is kept (the walls' axes)."""
        tr = [t for t in self.bowls if t.n >= P.RELOC_MIN_SIGHTINGS]
        if len(seen) < 2 or len(tr) < 2:
            return None
        S = np.array([b.centre[:2] for b in seen])
        T = np.array([t.centre[:2] for t in tr])
        tol = 0.2

        def score(s: np.ndarray) -> int:
            d = np.linalg.norm((S - s)[:, None, :] - T[None], axis=2)
            j = np.argmin(d, axis=1)
            ok = d[np.arange(len(S)), j] < tol
            return len(set(j[ok].tolist())) if ok.sum() >= 2 else 0

        if score(np.zeros(2)) >= 2:
            return None
        cands = [S[i] - T[j] for i in range(len(S)) for j in range(len(T))]
        scores = [score(c) for c in cands]
        best = max(scores)
        if best < 2:
            return None
        top = [c for c, s in zip(cands, scores) if s == best]
        if any(np.linalg.norm(c - top[0]) > 0.3 for c in top):
            return None
        s = np.mean(top, axis=0)
        if np.linalg.norm(s) < 0.3 or np.linalg.norm(s) > 6.:
            return None
        return s

    def heading(self, pts_r: np.ndarray) -> None:
        """Absolute heading from the house's walls and fittings, which run along two perpendicular directions: the angle `phi` (mod 90
        degrees) of those axes in the robot frame is where the head's 2D points (LOC_BAND_M heights) pile up most sharply when projected
        on them. The first views (the base stands at the start) fix the axes in the odometry frame (psi0); later views give the yaw psi0 -
        phi + k * 90 degrees, k nearest the odometry. The odometry under-counts turns by 3-7 per cent (native scans 301-303) and the error
        would grow with every turn; the axes give the yaw to 0.001-0.018 rad on every standing view of those scans."""
        self.loc_step = self.step
        z, r = pts_r[:, 2], np.hypot(pts_r[:, 0], pts_r[:, 1])
        q = pts_r[(z > P.LOC_BAND_M[0]) & (z < P.LOC_BAND_M[1]) & (r < P.LOC_RANGE_M)][:, :2]
        if len(q) < P.LOC_MIN_POINTS:
            return
        q = q[::len(q) // P.LOC_MAX_POINTS + 1]
        step = np.radians(P.AXES_STEP_DEG)
        phis = np.arange(0., np.pi / 2, step)
        sc = np.zeros(len(phis))
        for i, f in enumerate(phis):
            for w in (q[:, 0] * np.cos(f) + q[:, 1] * np.sin(f), -q[:, 0] * np.sin(f) + q[:, 1] * np.cos(f)):
                h = np.bincount(np.floor((w - w.min()) / P.AXES_BIN_M).astype(int)).astype(float)
                sc[i] += float(h @ h)
        k = int(np.argmax(sc))
        peak = float(sc[k] / max(np.median(sc), 1e-9))
        a, b, c = sc[k - 1], sc[k], sc[(k + 1) % len(sc)]
        den = a - 2. * b + c
        phi = phis[k] + (0.5 * (a - c) / den * step if den < 0 else 0.)
        yaw = float(self.odo.pose[2])
        self.last_loc = dict(step=self.step, phi=round(float(phi), 4), peak=round(peak, 2))
        if peak < P.AXES_MIN_PEAK:
            return
        quarter = np.pi / 2
        if self.psi0 is None or self.step <= P.AXES_FIX_STEPS:      # the base stands at the start: the axes in the odometry frame
            self.psi_votes.append(4. * ((yaw + phi) % quarter))
            self.psi0 = float(np.angle(np.mean(np.exp(1j * np.array(self.psi_votes)))) / 4.)
            return
        est = self.psi0 - phi
        est += quarter * round((yaw - est) / quarter)
        d = float(np.remainder(est - yaw + np.pi, 2 * np.pi) - np.pi)
        self.last_loc['dyaw'] = round(d, 4)
        if abs(d) > P.AXES_MAX_CORR:     # a large correction (a bump turned the base 0.33 rad unseen, t20f15) only when two views agree
            agree = self.big_corr is not None and abs(d - self.big_corr) < P.AXES_AGREE_RAD
            self.big_corr = d
            if not agree:
                return
        self.big_corr = None
        self.counts['loc'] = self.counts.get('loc', 0) + 1
        self.loc_total[2] += d
        self.odo.pose[2] = yaw + d

    # ---- products ------------------------------------------------------------------------------------------------------------------
    def world(self, q: np.ndarray) -> World:
        return World(step=self.step, base=self.pose(), bowls=list(self.bowls), vegs=list(self.vegs), baskets=list(self.baskets),
                     counter_top=P.COUNTER_TOP_M, cloud=self.low_cloud)

    def est(self) -> Dict[str, Any]:
        """Estimates for the trace (the harness compares them with the truth offline; never fed back)."""
        return dict(base=np.round(self.pose(), 3).tolist(), loc=dict(total=np.round(self.loc_total, 3).tolist(), last=self.last_loc),
                    bowls=[np.round(b.centre, 3).tolist() + [b.n] for b in self.bowls],
                    baskets=[np.round(b.centre, 3).tolist() + [round(b.yaw, 2), b.n] for b in self.baskets],
                    vegs=[np.round(v.centre, 3).tolist() + [v.kind, v.group, v.n_points] for v in self.vegs])
