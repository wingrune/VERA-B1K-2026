"""Pinch planning of the fair planner of task 35: top pinches of the fitted camera model reachable from stands (GPU reachable_stands mask),
their costs (stand, side, site, tilt), and the clearance checks against the fitted tripod (the fingers' approach line,
the arm's joint path to the pre-grasp point)."""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from .. import ik
from .. import models as MD
from .. import params as P


def _leg_segments(w):
    """(A, B) segment ends (K, 3) of the estimated tripod: apex under the head to every foot vertex, and the column."""
    if w.tripod is None:
        return None
    T = w.tripod.T
    F = (T @ MD.TRIPOD_F)[:3, 3]
    apex = F - np.array([0., 0., 0.15])
    feet = MD.TRIPOD_FEET @ T[:3, :3].T + T[:3, 3]
    a = np.vstack([np.tile(apex, (len(feet), 1)), apex[None]])
    b = np.vstack([feet, np.array([[apex[0], apex[1], w.floor_z + 0.40]])])
    return a, b


def approach_clear(w, opt) -> bool:
    """The fingers' approach (pre-grasp point to the pinch, and the wrist 10 cm behind) keeps LEG_CLEAR_M from the tripod legs."""
    seg = _leg_segments(w)
    if seg is None:
        return True
    a, b = seg
    pts = np.array([opt['pre'] + (opt['mid'] - opt['pre']) * t for t in np.linspace(0., 1., 9)])
    pts = np.vstack([pts, pts - opt['axis'] * 0.10])
    ab = b - a
    t = np.clip(np.einsum('nkd,kd->nk', pts[:, None, :] - a[None], ab) / np.maximum(np.einsum('kd,kd->k', ab, ab), 1e-12), 0., 1.)
    d = np.linalg.norm(pts[:, None, :] - (a[None] + t[..., None] * ab[None]), axis=-1)
    return bool(np.all(d > P.LEG_CLEAR_M))


PATH_LINKS = ('arm_link3', 'arm_link4', 'arm_link5', 'arm_link6', 'arm_link7', 'gripper_link')


def arm_path_clear(c, w, q, x) -> bool:
    """The straight joint path from the current joints to the servo target x (trunk 4 + arm 7, from solve_near) keeps the arm links
    LEG_CLEAR_M from the estimated tripod legs, column and head (forward kinematics at PATH_SAMPLES points)."""
    seg = _leg_segments(w)
    if seg is None:
        return True
    a, b = seg
    ab = b - a
    dofs = list(K.TRUNK_DOFS) + list(K.ARM_DOFS[c.side])
    x0 = q[dofs].copy()
    trial = np.zeros(64)
    trial[:len(q)] = q
    cs, sn = math.cos(w.base[2]), math.sin(w.base[2])
    names = [f'{c.side}_{n}' for n in PATH_LINKS]
    for lam in np.linspace(0., 1., P.PATH_SAMPLES + 1)[1:]:
        trial[dofs] = x0 + lam * (np.asarray(x, float) - x0)
        poses = K.kinematics().forward(trial)[0]
        pts = np.array([poses[n][:3, 3] for n in names])
        pts = np.vstack([pts, (pts[:-1] + pts[1:]) / 2])
        wp = np.c_[w.base[0] + cs * pts[:, 0] - sn * pts[:, 1], w.base[1] + sn * pts[:, 0] + cs * pts[:, 1], pts[:, 2]]
        t = np.clip(np.einsum('nkd,kd->nk', wp[:, None, :] - a[None], ab) / np.maximum(np.einsum('kd,kd->k', ab, ab), 1e-12), 0., 1.)
        d = np.linalg.norm(wp[:, None, :] - (a[None] + t[..., None] * ab[None]), axis=-1)
        if np.any(d < P.LEG_CLEAR_M):
            return False
    return True


def candidates(c, w, q, stands=None, ecost=None, penalty=None) -> List[Dict[str, Any]]:
    if stands is None:
        stands, ecost, penalty = c.stands_around(w, w.camera.xy, P.GRASP_RINGS)
    penalty = np.zeros(len(stands)) if penalty is None else penalty
    opts = G.grasp_options(w.camera.T, MD.CAMERA_VERTS, P.TILTS)
    out = []
    if not opts:
        return out
    for side in ('left', 'right'):
        goals = []
        for o in opts:
            goals += [G.pose_T(o['rotation'], o['pre']), G.pose_T(o['rotation'], o['mid'])]
        masks, _ = ik.scan(side, goals, stands, 0., K.FINGER_MID_POINT)     # reachability mask only
        for j, o in enumerate(opts):
            for i in np.flatnonzero(masks[2 * j] & masks[2 * j + 1]):
                key = (o['site'], o['sign'], round(o['tilt'], 2), o['lean'], side, tuple(np.round(stands[i][:2], 1)))
                if key in c.bad_grasp:
                    continue
                out.append(dict(stand=stands[i].copy(), side=side, opt=o, key=key, ecost=float(ecost[i]),
                                cost=float(ecost[i]) + float(penalty[i]) + float(o['cost']) + P.SIDE_COST[side]))
    return out


def check(w):
    def ok(cand):
        return approach_clear(w, cand['opt'])
    return ok
