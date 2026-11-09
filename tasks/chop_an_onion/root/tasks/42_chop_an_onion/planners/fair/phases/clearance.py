"""Collision check of an arm configuration against the counter: the palm, the forearm and the fingers as spheres, the counter as a slab with the
basin cut out of it (all from the perceived counter and basin). The reach scan of cuRobo says where the hand can be, not what the arm touches:
an onion 6 cm behind the basin's front wall cannot be taken with the palm straight above it, the palm lands on the rim."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from core.skills_lib import r1pro_kinematics as K

from .. import params as P
from ..world import World

PALM_R, ARM_R, FINGER_R = 0.045, 0.04, 0.012


def link_spheres(side: str, q11: np.ndarray) -> List[Tuple[np.ndarray, float]]:
    """Arm collision spheres.
    (centre in the robot frame, radius) of the palm, the forearm and the fingers for the trunk + arm joints q11 (open fingers)."""
    q = np.zeros(22)
    q[list(K.TRUNK_DOFS) + list(K.ARM_DOFS[side])] = q11
    q[list(K.FINGER_DOFS[side])] = 0.05
    poses = K.kinematics().forward(q)[0]
    out: List[Tuple[np.ndarray, float]] = []
    chain = [poses[f'{side}_arm_link{i}'][:3, 3] for i in (4, 5, 6, 7)]
    g = poses[f'{side}_gripper_link']
    for a, b in zip(chain[:-1], chain[1:]):
        for t in np.linspace(0., 1., 4):
            out.append((a + (b - a) * t, ARM_R))
    palm = g[:3, 3]
    out.append((palm, PALM_R))
    out.append((0.5 * (chain[-1] + palm), ARM_R))
    tip = g[:3, 3] + g[:3, :3] @ np.array([0., 0., -0.085])
    for t in np.linspace(0.3, 1., 4):
        out.append((palm + (tip - palm) * t, FINGER_R))
    return out


def hits(w: World, side: str, q11: np.ndarray, stand: np.ndarray, why: list = None) -> bool:
    """Arm sphere inside the counter slab.
    True when a sphere of the arm at the joints q11, the base standing at `stand` (x, y, yaw), is inside the counter slab (not over the basin)."""
    ce, ba = w.counter, w.basin
    if ce is None:
        return False
    c, s = np.cos(stand[2]), np.sin(stand[2])
    for p, r in link_spheres(side, q11):
        xy = stand[:2] + np.array([c * p[0] - s * p[1], s * p[0] + c * p[1]])
        a, b = ce.to_local(xy)
        if abs(a) > 0.5 * ce.length + r or abs(b) > 0.5 * ce.depth + r:
            continue
        if p[2] >= ce.top + r - P.CLEAR_SLACK_M:
            continue
        if ba is not None:
            ba_, bb_ = ce.to_local(ba.centre)
            if abs(a - ba_) < 0.5 * ba.size[0] - P.CLEAR_BASIN_SHRINK_M and abs(b - bb_) < 0.5 * ba.size[1] - P.CLEAR_BASIN_SHRINK_M:
                if p[2] >= ba.floor + r - P.CLEAR_SLACK_M:
                    continue
        if why is not None:
            why.append(dict(r=r, a=round(float(a), 3), b=round(float(b), 3), z=round(float(p[2]), 3), top=round(float(ce.top), 3),
                            basin=None if ba is None else [round(float(v), 3) for v in ce.to_local(ba.centre)], D=round(float(ce.depth), 3)))
        return True
    return False
