"""Carry phases of the fair planner of task 35: tuck the held camera high in front of the chest (with the orientation it will have at the
mount), find the tripod when it is not known yet, choose a stand that reaches the pose over the mount for the current tripod yaw
hypothesis (GPU reachable_stands mask over rings around F), walk there, refit the tripod from the stand. Subgoal near_tripod."""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from .. import ik
from .. import models as MD
from .. import params as P
from .grasp import drop


def attempts(w) -> List[tuple]:
    """Mount attempts (tripod yaw hypothesis, position dither index, yaw dither index): every hypothesis straight first, then the yaw
    dithers of each (the camera's yaw in the hand and the tripod's yaw add up against the 15 degree limit), then the position dithers."""
    n = max(1, len(w.tripod.yaws)) if w.tripod is not None else 1
    return ([(h, 0, 0) for h in range(n)] + [(h, 0, k) for h in range(n) for k in range(1, len(P.DITHER_YAW))]
            + [(h, d, 0) for h in range(n) for d in range(1, len(P.DITHER))])


def f_T(c, w) -> np.ndarray:
    """F frame (odometry) of the tripod under the current yaw hypothesis; a guess 1 m ahead at the mount height when no tripod is known."""
    if w.tripod is None:
        b = w.base
        T = G.planar_T(b[0] + math.cos(b[2]), b[1] + math.sin(b[2]), w.floor_z + MD.TRIPOD_FLOOR, b[2])
        return T @ MD.TRIPOD_F
    seq = attempts(w)
    h, _, k = seq[min(c.attempt, len(seq) - 1)]
    yaws = w.tripod.yaws or [G.yaw_of(w.tripod.T)]
    T = G.planar_T(w.tripod.T[0, 3], w.tripod.T[1, 3], w.tripod.T[2, 3], yaws[min(h, len(yaws) - 1)] + P.DITHER_YAW[k])
    return T @ MD.TRIPOD_F


def mid_in_cam(c) -> np.ndarray:
    return np.linalg.inv(c.perc.held['T_mid_cam'])


def mount_T(c, w, dz: float, dither: bool = True) -> np.ndarray:
    T = G.raised(G.mount_goal(f_T(c, w), MD.CAMERA_M, mid_in_cam(c)), dz)
    if dither and w.tripod is not None:
        seq = attempts(w)
        dx, dy = P.DITHER[seq[min(c.attempt, len(seq) - 1)][1]]
        T[0, 3] += dx
        T[1, 3] += dy
    return T


def tuck_goal(c, w):
    T = mount_T(c, w, P.PRE_M, dither=False)
    z = T[2, 3] + P.TUCK_UP
    y = P.TUCK_Y if c.side == 'left' else -P.TUCK_Y      # beside the head camera's line of sight to the mount
    F = f_T(c, w)
    face = math.atan2(F[1, 3] - w.base[1], F[0, 3] - w.base[0])
    return np.array([P.TUCK_X, y, z]), G.rot_yaw(-face) @ T[:3, :3]


def tuck(c, action, q, w) -> None:
    if not c.holds(q, c.side):
        drop(c, w, 'camera lost while tucking')
        return
    c.other_rest(action, q)
    if c.age == 1:
        goal, rot = tuck_goal(c, w)
        x = c.servo.solve(q, goal, rot)
        if x is None:          # keep the lift orientation when the mount orientation is out of reach here
            rot = G.rotation_to_robot(w.base, c.rot_hold)
            x = c.servo.solve(q, goal, rot)
        c.tuck = (goal, rot)
        c.servo.target = x
        c.servo.sync = True
        c.reason = f'tuck target {"found" if x is not None else "not found, tracking"}'
    goal, rot = c.tuck
    c.err = c.servo.step(action, q, goal, rot)
    if (c.err[0] < 0.03 and c.err[1] < 0.2 and c.servo.target is None) or c.age > P.REACH_TIMEOUT:
        c.hold_q = q[list(K.TRUNK_DOFS) + list(K.ARM_DOFS[c.side])].copy()
        c.turned, c.scan_from = 0., None
        c.change('find_tripod' if w.tripod is None else 'plan_mount', f'camera tucked, err {c.err[0]:.3f} m {c.err[1]:.2f} rad')


def find_tripod(c, action, q, w) -> None:
    if not c.holds(q, c.side):
        drop(c, w, 'camera lost while looking for the tripod')
        return
    c.hold_arm(action)
    c.other_rest(action, q)
    if c.scan_from is not None:
        c.turned += abs(math.atan2(math.sin(w.base[2] - c.scan_from), math.cos(w.base[2] - c.scan_from)))
    c.scan_from = float(w.base[2])
    action[2] = P.SCAN_RATE * 0.6
    c.reason = f'turning with the camera for the tripod, {c.turned:.2f} rad'
    if w.tripod is not None and c.age > P.LOOK_STEPS:
        if P.RETUCK:          # the tuck orientation was a guess without the tripod: tuck again with the mount orientation
            c.servo = ik.make_servo(c.side, 'mid')
            c.change('tuck', 'tripod found; tucking again with its mount orientation')
        else:
            c.change('plan_mount', 'tripod found')
    elif c.turned > P.SCAN_FULL_TURN_RAD * 1.2:
        c.change('failed', 'tripod not found')


def mount_candidates(c, w, q) -> List[Dict[str, Any]]:
    T = mount_T(c, w, 0., dither=False)
    pre = G.raised(T, P.PRE_M)
    F = f_T(c, w)
    stands, ecost, penalty = c.stands_around(w, F[:3, 3], P.MOUNT_RINGS)
    masks, _ = ik.scan(c.side, [pre, T], stands, 0., K.FINGER_MID_POINT)      # reachability mask only
    out = []
    for i in np.flatnonzero(masks.all(0)):
        key = tuple(np.round(stands[i], 1))
        if key in c.bad_mount:
            continue
        out.append(dict(stand=stands[i].copy(), side=c.side, ecost=float(ecost[i]), cost=float(ecost[i] + penalty[i]), key=key))
    return out


def plan_mount(c, action, q, w) -> None:
    if not c.holds(q, c.side):
        drop(c, w, 'camera lost before the mount')
        return
    c.hold_arm(action)
    if c.counts['mount_rounds'] >= P.MAX_MOUNT_ROUNDS + len(attempts(w)):
        c.change('release', 'mount rounds exhausted; letting go over the tripod')
        return
    c.counts['mount_rounds'] += 1
    c.build_grid(w, [f_T(c, w)[:3, 3]])
    cands = mount_candidates(c, w, q)
    c.mount = c.pick(w, cands)
    if c.mount is None:
        if c.attempt + 1 < len(attempts(w)):
            c.attempt += 1
            c.change('plan_mount', f'no stand reaches the mount for this tripod yaw {c.pick_stats}; next hypothesis')
            return
        c.change('failed', f'no stand reaches the mount {c.pick_stats}')
        return
    c.servo = None
    c.change('carry', f"to stand {np.round(c.mount['stand'], 2).tolist()}, route cost {c.mount['total']:.2f}, attempt {c.attempt}")


def carry(c, action, q, w) -> None:
    if not c.holds(q, c.side):
        drop(c, w, 'camera lost while carrying')
        return
    c.hold_arm(action)
    c.other_rest(action, q)
    g = c.mount['stand']
    dist, yaw_err = c.navigate(action, q, w, g[:2], g[2], rest=False, vmax=P.CARRY_VMAX, wmax=P.CARRY_WMAX)
    c.hold_arm(action)
    c.err = (dist, yaw_err)
    if c.arrived(dist, yaw_err):
        c.change('look_mount', f'at the mount stand, {dist:.3f} m, yaw err {yaw_err:.2f}')
    elif c.counts['nav_stuck'] >= P.NAV_MAX_STUCK or c.age > P.NAV_TIMEOUT:
        c.bad_mount.add(c.mount['key'])
        c.change('plan_mount', f'mount stand not reached ({dist:.2f} m)')


def look_mount(c, action, q, w) -> None:
    """Stand still, forget the tripod estimate and refit it from the stand (closer view, the handle visible, consistent with the odometry
    pose the arm will use), then freeze it."""
    if not c.holds(q, c.side):
        drop(c, w, 'camera lost at the mount stand')
        return
    c.hold_arm(action)
    c.other_rest(action, q)
    if c.age == 1:
        c.perc.refresh('tripod')
    if c.age < P.LOOK_STEPS or (w.tripod is None and c.age < 3 * P.LOOK_STEPS):
        return
    if w.tripod is None:
        c.turned, c.scan_from = 0., None
        c.change('find_tripod', 'tripod not seen from the mount stand')
        return
    c.perc.freeze['tripod'] = True
    c.servo = ik.make_servo(c.side, 'mid')
    c.change('to_pre_mount', f'tripod refitted, yaw hypotheses {[round(v, 2) for v in w.tripod.yaws]}')


PHASES = dict(tuck=tuck, find_tripod=find_tripod, plan_mount=plan_mount, carry=carry, look_mount=look_mount)
