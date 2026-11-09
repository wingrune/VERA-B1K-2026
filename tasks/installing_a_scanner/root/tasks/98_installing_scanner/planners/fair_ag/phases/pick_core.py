"""Shared helpers of the pick and place phases of the fair planner: base stands on rings around a point (one batched GPU IK scan over all of them,
A* routes on the perceived grid only for the cheapest), the joint path check of an arm against the perceived desk, laptop and scanner, the hold
of the native assisted grasp read from the finger joints (proprio only), and the finger midpoint pose by forward kinematics. Plans keep the
state of a pick in `c.pick` (a dict reset with the planner)."""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from core.skills_lib import nav_grid
from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from ..world import Box

PATH_LINKS = ('arm_link3', 'arm_link4', 'arm_link5', 'arm_link6', 'arm_link7', 'gripper_link', 'gripper_finger_link1', 'gripper_finger_link2')
PICK_RINGS = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05)   # stand rings around the scanner or the place (m)
PICK_BEARINGS, PICK_YAW_OFFSETS, MAX_STANDS, ROUTE_CHECKS = 24, (0., 0.3, -0.3), 300, 20
PATH_SAMPLES, PATH_GROW_M = 20, 0.02     # arm joint path check: samples, growth of the perceived boxes
HOLD_MIN_GAP_M, HOLD_LOST_GAP_M, HOLD_STILL_M = 0.004, 0.002, 0.0005   # finger gap (sum of both joints) of a hand closed on the scanner, of an empty one


def new_pick() -> Dict[str, Any]:
    return dict(tries=0, fails=0, drops=0, place_rounds=0, bad=set(), cands=[], grasp=None, world=None, rel=None, hold_q=None, lift0=None,
                lift1=None, base1=None, rot=None, carry=None, held_age=0, gaps=[], retreat=None, placed=False)


def stands_around(c, w, centre) -> Tuple[np.ndarray, np.ndarray]:
    """Stands around a point.
    Base poses (N, 3) on rings around `centre` facing it, free in the inflated grid, cheapest first by straight-line cost; the current base
    pose is the first one (cost 0)."""
    rows = [(0., w.base[0], w.base[1], w.base[2])]
    ctr = np.asarray(centre, float)[:2]
    for r in PICK_RINGS:
        for b in np.arange(PICK_BEARINGS) * 2 * math.pi / PICK_BEARINGS:
            xy = ctr + r * np.array([math.cos(b), math.sin(b)])
            if not c.free_pose(xy):
                continue
            for dy in PICK_YAW_OFFSETS:
                yaw = nav_grid.wrap(b + math.pi + dy)
                rows.append((float(np.linalg.norm(xy - w.base[:2])) + 0.15 * abs(nav_grid.wrap(yaw - w.base[2])), xy[0], xy[1], yaw))
    rows = [rows[0]] + sorted(rows[1:])[:MAX_STANDS - 1]
    arr = np.array(rows, float)
    return arr[:, 1:], arr[:, 0]


def route_len(c, w, xy) -> float:
    if float(np.linalg.norm(xy - w.base[:2])) < 0.03:
        return 0.
    path = c.inflated.path(w.base[:2], xy)
    return float('inf') if path is None else nav_grid.path_length(path)


def pick_routed(c, w, cands, path_check=None):
    """Cheapest routed candidate.
    By route length plus its own cost terms; only the ROUTE_CHECKS cheapest (by straight-line cost) that pass
    `path_check` are routed. The chosen one is removed from `cands`."""
    cands.sort(key=lambda d: d['cost'])
    best, routed = None, 0
    for cand in cands:
        if routed >= ROUTE_CHECKS:
            break
        if path_check is not None and not path_check(cand):
            continue
        routed += 1
        d = route_len(c, w, cand['stand'][:2])
        if math.isfinite(d) and (best is None or cand['cost'] - cand['ecost'] + d < best[0]):
            best = (cand['cost'] - cand['ecost'] + d, cand)
    if best is None:
        return None
    cands[:] = [d for d in cands if d is not best[1]]
    best[1]['total'] = best[0]
    return best[1]


def arm_path_clear(w, side, q, x0, x1, base, scanner=True) -> bool:
    """Arm joint path clear of the boxes.
    The joint path the servo walks from x0 to x1 (trunk + arm, all joints together) keeps the links, the midpoints between them and the
    fingertips out of the perceived desk (its box reaches the floor), the laptop and, with `scanner`, the oriented scanner box."""
    grow = np.full(3, PATH_GROW_M)
    boxes = [Box(b.lo - grow, b.hi + grow) for b in (w.table_box, w.laptop_box)]
    sc = w.scanner
    lo = np.array([-sc.half[0], -sc.half[1], 0.]) - grow
    hi = np.array([sc.half[0], sc.half[1], sc.z_hi - sc.z_lo]) + grow
    rot, origin = G.rot_yaw(sc.yaw), np.array([sc.centre[0], sc.centre[1], sc.z_lo])
    dofs = list(K.TRUNK_DOFS) + list(K.ARM_DOFS[side])
    cs, sn = math.cos(base[2]), math.sin(base[2])
    trial = q.copy()
    names = [f'{side}_{n}' for n in PATH_LINKS]
    for a in np.linspace(0., 1., PATH_SAMPLES + 1)[1:]:
        trial[dofs] = x0 + a * (x1 - x0)
        poses = K.kinematics().forward(trial)[0]
        pts = np.array([poses[n][:3, 3] for n in names])
        tips = [poses[f'{side}_{f}'][:3, :3] @ K.TIP_POINT + poses[f'{side}_{f}'][:3, 3] for f in ('gripper_finger_link1', 'gripper_finger_link2')]
        pts = np.vstack([pts, (pts[:-2] + pts[1:-1]) / 2, tips])
        wp = np.c_[base[0] + cs * pts[:, 0] - sn * pts[:, 1], base[1] + sn * pts[:, 0] + cs * pts[:, 1], pts[:, 2]]
        if any(np.any(np.all((wp >= b.lo) & (wp <= b.hi), axis=1)) for b in boxes):
            return False
        if scanner and np.any(np.all(((wp - origin) @ rot >= lo) & ((wp - origin) @ rot <= hi), axis=1)):
            return False
    return True


def finger_gap(q, side) -> float:
    """Hand opening.
    Sum of both finger joints, 0 closed, 0.1 fully open."""
    return float(np.sum(np.abs(q[list(K.FINGER_DOFS[side])])))


def update_hold(c, q) -> bool:
    """Hold of the closing hand.
    While the hand is commanded closed: it holds the scanner when the fingers stopped on it (gap above HOLD_MIN_GAP_M and still over 5
    controls). The native assisted grasp only fixes an object the closing fingers rest on, so an empty hand closes fully."""
    gaps = c.pick['gaps']
    gaps.append(finger_gap(q, c.side))
    del gaps[:-6]
    return len(gaps) == 6 and gaps[-1] > HOLD_MIN_GAP_M and abs(gaps[-1] - gaps[0]) < HOLD_STILL_M


def holding(c, q) -> bool:
    return finger_gap(q, c.side) > HOLD_LOST_GAP_M


def mid_pose(q, w, side) -> np.ndarray:
    """Hand pose in the desk frame.
    Finger midpoint and gripper-link axes of `side` (4x4), from the joints and the base estimate."""
    servo = K.ArmServo(side)
    servo.use_fingers_mid()
    t = servo.pose(q)
    T = np.eye(4)
    rb = G.rot_yaw(w.base[2])
    T[:3, :3] = rb @ t[:3, :3]
    T[:3, 3] = np.r_[w.base[:2], 0.] + rb @ t[:3, 3]
    return T


def hold_arm(c, action) -> None:
    hq = c.pick['hold_q']
    action[K.TRUNK_ACTION] = hq[:4]
    action[K.ARM_ACTION[c.side]] = hq[4:]
