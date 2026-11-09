"""Geometry of the task 69 fair planner: the handle pinch over the ToggledOn marker, press faces, base stands, the vacuum head footprint, frame
helpers. The planner's world is the odometry frame (z from the robot frame origin).

Vacuum body (base_link) frame, from the asset prior (models.py): the top handle loop is a collision box (HANDLE_LO..HANDLE_HI, 10.6 x 4 x
19 cm), the ToggledOn marker (overlap radius 1.87 cm) sits inside it, 2.3 cm in from the +x face on the loop's mid plane; the projection cube
(30 x 12 x 2 cm, long side along body y) lies under the head.

Gripper conventions (as the task 69 oracle): gripper link columns [x, y, z] in the world, y the closing axis, z = -approach; goals are
finger-midpoint poses (K.FINGER_MID_POINT).
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

from . import models as MD

UP = np.array([0., 0., 1.])
PRE_GAP_M = 0.12
P_ROLLS = (0., math.pi / 2, math.pi / 4, -math.pi / 4, 3 * math.pi / 4, -3 * math.pi / 4)   # turns of the closed fingers about the push axis tried per face
HEAD_HALF = (0.17, 0.15)      # vacuum head half extents (m) along the cube's long axis and across it, around the cube centre


def unit(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def rot_yaw(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def pose_T(rotation: np.ndarray, point: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rotation, point
    return T


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def yaw_of(T: np.ndarray) -> float:
    return math.atan2(T[1, 0], T[0, 0])


def planar_T(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    return pose_T(rot_yaw(yaw), np.array([x, y, z]))


def to_base(base: Sequence[float], p_world: Sequence[float]) -> np.ndarray:
    """World xy point -> base frame xy."""
    c, s = math.cos(base[2]), math.sin(base[2])
    d = np.asarray(p_world, float)[:2] - np.asarray(base[:2], float)
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])



def standing_T(x, y, yaw, floor_z) -> np.ndarray:
    """Pose of the vacuum body frame standing on the floor at.

    Pose of the vacuum body frame standing on the floor at (x, y) with the given yaw."""
    return planar_T(x, y, floor_z + MD.FLOOR_Z, yaw)


def handle_point(deep: float) -> np.ndarray:
    """Pinch point on the loop's -x side in the body frame.

    Pinch point on the loop's -x side in the body frame: `deep` inside the -x face, on the mid plane, at the marker height."""
    return np.array([MD.HANDLE_LO[0] + deep, 0.5 * (MD.HANDLE_LO[1] + MD.HANDLE_HI[1]), MD.MARKER[2]])


def pinch_frames(vac_T, point_body, pitches, from_side=-1.0, yaws=(0.,), rolls=(0.,)) -> List[Dict[str, object]]:
    """Grasp frames on the handle.

    World finger-midpoint grasp frames on the handle: the hand moves along body +x from the -x side of the loop (from_side -1) or along -x
    from the +x side (from_side +1), pitched down by each `pitches` angle (rad) in the body x-z plane and turned about the body vertical by
    each of `yaws`, fingers closing along body y (both signs) turned about the approach axis by each of `rolls`."""
    rv, tv = vac_T[:3, :3], vac_T[:3, 3]
    out = []
    for pitch in pitches:
        for yaw in yaws:
            a_body = rot_yaw(yaw) @ np.array([-from_side * math.cos(pitch), 0., -math.sin(pitch)])
            a = unit(rv @ a_body)
            point = tv + rv @ np.asarray(point_body, float)
            for sign in (1., -1.):
                z = -a
                y = sign * rv[:, 1]
                y = unit(y - float(y @ z) * z)
                x = np.cross(y, z)
                base_rot = np.column_stack([x, y, z])
                for roll in rolls:
                    out.append(dict(sign=sign, pitch=float(pitch), yaw=float(yaw), roll=float(roll), point=point.copy(), rot=base_rot @ rot_z_local(roll),
                                    pre=point - a * PRE_GAP_M, approach=a, from_side=float(from_side)))
    return out


def rot_z_local(angle: float) -> np.ndarray:
    """Rotation about the local z axis.

    Rotation about the local z axis (the gripper's approach axis) by `angle`."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def press_frame(vac_T, point_body, normal_body, pre_m=0.10, close_body=(0., 0., 1.)) -> Dict[str, object]:
    """Press frame on a handle face.

    World frame of a closed-gripper press on a handle face: the finger midpoint (with closed fingers, the fingertip pad) moves along
    -normal onto `point_body` (a point on the face, body frame) from `pre_m` outside (negative: inside). The fingers point into the face
    (gripper z = the outward normal); the closed fingers stack along `close_body`, made orthogonal to the normal."""
    rv, tv = vac_T[:3, :3], vac_T[:3, 3]
    n = unit(rv @ np.asarray(normal_body, float))
    z = n
    c = rv @ np.asarray(close_body, float)
    y = c - float(c @ z) * z
    if np.linalg.norm(y) < 1e-6:
        y = rv[:, 0] - float(rv[:, 0] @ z) * z
    y = unit(y)
    x = np.cross(y, z)
    point = tv + rv @ np.asarray(point_body, float)
    return dict(point=point, rot=np.column_stack([x, y, z]), normal=n, pre=point + n * pre_m)


def press_faces(touch, press_in) -> List[Dict[str, object]]:
    """Press targets on the handle loop, several finger turns each.

    The press targets on the handle loop (body frame): faces y+, y-, x+ at the marker height and x; each with the closed fingers turned about
    the push axis by several angles (the finger roll does not matter for a push, the arm's reach does)."""
    lo, hi = MD.HANDLE_LO, MD.HANDLE_HI
    xm, zm, ym = MD.MARKER[0], MD.MARKER[2], 0.5 * (lo[1] + hi[1])
    faces = []
    for name, point, normal, alt in (('y+', [xm, hi[1], zm], [0., 1., 0.], [1., 0., 0.]),
                                     ('y-', [xm, lo[1], zm], [0., -1., 0.], [1., 0., 0.]),
                                     ('x+', [hi[0], ym, zm], [1., 0., 0.], [0., 1., 0.])):
        n = np.array(normal, float)
        for k, ang in enumerate(P_ROLLS):
            c0 = np.array([0., 0., 1.])
            c = c0 * math.cos(ang) + np.cross(n, c0) * math.sin(ang)
            faces.append(dict(name=f'{name}{k}', point=np.array(point), normal=n, close=c, touch=touch, final=-press_in))
    return faces


def ring_stands(centre_xy, radii, n_bearings, yaw_offsets) -> np.ndarray:
    """Base poses (N, 3) on rings around `centre_xy`.

    Base poses (N, 3) on rings around `centre_xy`, facing it with yaw offsets."""
    rows = []
    for r in radii:
        for k in range(n_bearings):
            th = 2 * math.pi * k / n_bearings
            xy = np.asarray(centre_xy, float) + r * np.array([math.cos(th), math.sin(th)])
            face = math.atan2(centre_xy[1] - xy[1], centre_xy[0] - xy[0])
            for off in yaw_offsets:
                rows.append([xy[0], xy[1], wrap(face + off)])
    return np.asarray(rows, float).reshape(-1, 3)



def order_greedy(start, points) -> List[int]:
    """Visiting order of the particles.

    Visiting order of the particles: nearest neighbour from `start` (xy), then 2-opt on the open path."""
    left = dict(points)
    cur = np.asarray(start, float)[:2]
    order = []
    while left:
        pid = min(left, key=lambda k: float(np.linalg.norm(left[k][:2] - cur)))
        order.append(pid)
        cur = left.pop(pid)[:2]
    pts = [np.asarray(start, float)[:2]] + [np.asarray(points[k], float)[:2] for k in order]

    def length(seq):
        return sum(float(np.linalg.norm(seq[i + 1] - seq[i])) for i in range(len(seq) - 1))

    best = length(pts)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(pts) - 1):
            for j in range(i + 1, len(pts)):
                trial = pts[:i] + pts[i:j + 1][::-1] + pts[j + 1:]
                cand = length(trial)
                if cand < best - 1e-6:
                    pts, best, improved = trial, cand, True
                    order = order[:i - 1] + order[i - 1:j][::-1] + order[j:]
    return order
