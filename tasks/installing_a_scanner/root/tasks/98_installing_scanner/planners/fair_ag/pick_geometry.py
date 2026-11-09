"""Pick and place geometry of the task 98 fair planner (numpy only), on the estimated World: the edge pinch of the scanner for the native
assisted grasp, the resting poses of the scanner next to the laptop, and the scanner pose as a 4x4 transform. The same plans as the privileged
planner of the task, but the scanner is the perceived box (no collision hull): its frame sits at the bottom centre of the box, +x out of the
button face, z up from the desk top. All poses are in the desk frame D of the World (z in the robot frame)."""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from . import geometry as G
from . import params as P
from .world import ScannerEst, World

# pinch sets in order of preference: jaw tilts from the vertical (deg), finger midpoint inset from the face and bite below the hull top (m),
# away: on the long faces only offsets away from the button (their tops slope to the edge, a pinch at the centre squeezes out)
PINCH_SETS = (((42.5, 45.), 0.020, 0.016, True), ((45., 30., 60.), 0.010, 0.008, False))
GRASP_OFFSETS_M = (0.0, 0.05, -0.05)  # pinch points along the edge from its centre
TOP_DROP_M = 0.012                 # hull top near the side faces below the box top (asset template; the ridge at the button face is the box top)
GRASP_BACK_M = 0.10                # pre-grasp point: backed off along the tool axis
EDGE_KEEP_M = 0.05                 # pinch points stay this far from the ends of an edge (and so from the button)
PLACE_GAP_M = 0.03                 # AABB gap scanner-laptop of the placed scanner (the NextTo bound is about 0.075)
PLACE_CLEAR_M = 0.003              # placed scanner: bottom this far above the desk top
QUARTER_COST = 0.1                 # cost of each quarter turn of the scanner yaw between pick and place
PRESS_FACE_COST, PRESS_REACH_COST, PRESS_REACH_M = 1.0, 2.0, 0.7   # carry before the press: button towards the place stand, within reach


def pose_T(rotation, point) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rotation, point
    return T


def scanner_T(sc: ScannerEst) -> np.ndarray:
    """Box pose: bottom centre, button yaw."""
    return pose_T(G.rot_yaw(sc.yaw), [sc.centre[0], sc.centre[1], sc.z_lo])


def scanner_from_T(T, like, step) -> ScannerEst:
    """Box `like` moved to the pose T.
    The scanner after a place, from the hand pose and the grasp transform."""
    return ScannerEst(centre=T[:2, 3].copy(), yaw=math.atan2(T[1, 0], T[0, 0]), half=tuple(like.half), z_lo=float(T[2, 3]),
                      z_hi=float(T[2, 3] + like.z_hi - like.z_lo), seen_step=step, front_known=like.front_known)


def marker_local(sc: ScannerEst) -> np.ndarray:
    """Button in the box frame.
    Template of the asset, see params."""
    return np.array([P.SCANNER_MARKER_FRONT_M, 0., sc.z_hi - sc.z_lo - P.SCANNER_MARKER_BELOW_TOP_M])


def edge_grasp(f, face, offset, theta, inset, bite):
    """Pinch of a top edge.
    The scanner lies flat (6 cm thick: too flat for a pinch across it, too wide for one across its top). The jaw lies in the vertical plane
    through the face normal n, tilted `theta` rad from the vertical: one finger comes down on the top face just inside the edge, the other
    closes on the side face just below it, so the rays between the fingers cross the corner (what the native assisted grasp tests). Fingers
    point along `axis` (inward and down); the finger midpoint is `inset` inside the face and `bite` below the hull top there. Returns the edge
    point, the finger midpoint, the pre-grasp point (GRASP_BACK_M back along the axis) and the gripper-link rotation."""
    n, centre, _ = f.faces()[face]
    t = G.unit(np.cross(G.UP, n))
    edge = centre + t * offset
    edge[2] = f.z_hi - TOP_DROP_M
    jaw = math.sin(theta) * n - math.cos(theta) * G.UP
    axis = -math.cos(theta) * n - math.sin(theta) * G.UP
    mid = edge - n * inset
    mid[2] = edge[2] - bite
    return dict(face=face, offset=float(offset), theta=float(theta), inset=inset, bite=bite, normal=n, tangent=t, edge=edge, axis=axis, jaw=jaw,
                mid=mid, pre=mid - axis * GRASP_BACK_M, rotation=G.jaw_rotation(axis, jaw))


def grasp_options(f) -> List[Dict[str, object]]:
    """Pinches of the non-button faces.
    Every PINCH_SETS entry at GRASP_OFFSETS_M along each edge, kept EDGE_KEEP_M from the edge ends; rank = preference order."""
    out, rank = [], 0
    for thetas, inset, bite, away in PINCH_SETS:
        for name, (n, _, half_w) in f.faces().items():
            if float(n @ f.front) > 0.5:
                continue
            for off in GRASP_OFFSETS_M:
                t = G.unit(np.cross(G.UP, n))
                if abs(off) > half_w - EDGE_KEEP_M or (away and abs(float(n @ f.front)) < 0.5 and float(t @ f.front) * off >= 0.):
                    continue
                for k, th in enumerate(thetas):
                    out.append(dict(edge_grasp(f, name, off, math.radians(th), inset, bite), rank=rank + k))
        rank += len(thetas)
    return out


def place_poses(w: World) -> List[Dict[str, object]]:
    """Resting poses next to the laptop.
    The current yaw plus 0-3 quarter turns: the scanner AABB PLACE_GAP_M from the laptop's along a desk axis, centred on the laptop across it
    (or as close to the current position as the laptop allows), inside the desk top by EDGE_MARGIN_M, the bottom PLACE_CLEAR_M above the desk
    top. Each: T (4x4 box pose), centre (AABB centre), yaw, quarter, cost."""
    sc, lb, tb = w.scanner, w.laptop_box, w.table_box
    a, b = sc.half
    h = sc.z_hi - sc.z_lo
    pts = np.array([[x, y, z] for x in (-a, a) for y in (-b, b) for z in (0., h)])
    c_now = w.scanner_box.centre
    out: List[Dict[str, object]] = []
    for quarter in range(4):
        yaw = sc.yaw + quarter * math.pi / 2
        p = (G.rot_yaw(yaw) @ pts.T).T
        plo, phi = p.min(0), p.max(0)
        size, off = phi - plo, (plo + phi) / 2
        for k in (0, 1):
            j = 1 - k
            for sign in (1., -1.):
                ck = lb.hi[k] + PLACE_GAP_M + size[k] / 2 if sign > 0 else lb.lo[k] - PLACE_GAP_M - size[k] / 2
                for cj in (lb.centre[j], float(np.clip(c_now[j], lb.lo[j], lb.hi[j]))):
                    centre = np.zeros(3)
                    centre[k], centre[j] = ck, cj
                    lo2, hi2 = centre[:2] - size[:2] / 2, centre[:2] + size[:2] / 2
                    if np.any(lo2 < tb.lo[:2] + P.EDGE_MARGIN_M) or np.any(hi2 > tb.hi[:2] - P.EDGE_MARGIN_M):
                        continue
                    if any(abs(o['yaw'] - yaw) < 1e-6 and np.linalg.norm(o['centre'][:2] - centre[:2]) < 0.01 for o in out):
                        continue
                    root = np.array([centre[0] - off[0], centre[1] - off[1], tb.hi[2] + PLACE_CLEAR_M - plo[2]])
                    centre[2] = root[2] + off[2]
                    out.append(dict(T=pose_T(G.rot_yaw(yaw), root), centre=centre, yaw=yaw, quarter=quarter, cost=QUARTER_COST * min(quarter, 4 - quarter)))
    return out


def press_after_place_cost(T, marker, stand) -> float:
    """Place before the press: button to the stand.
    Prefer the button facing the place stand and within reach of it."""
    m = (T @ np.r_[marker, 1.])[:3]
    d = stand[:2] - m[:2]
    dist = max(float(np.linalg.norm(d)), 1e-6)
    facing = float(T[:2, 0] @ d) / dist
    return PRESS_FACE_COST * (1. - facing) / 2. + PRESS_REACH_COST * max(0., dist - PRESS_REACH_M)
