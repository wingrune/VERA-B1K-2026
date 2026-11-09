"""Pure geometry of the task 98 fair planner (numpy only), on the estimated World: scanner frame and faces, button poke and push plans, tool
orientation, base candidates. Same plans as the oracle, but every input is an estimate from RGB-D and odometry.

Conventions: world frame, z up. A tool axis `a` is the unit direction from the wrist to the fingertip; the gripper link z axis is -a, its
x axis is horizontal (up x z_g) and y completes the frame (the same convention as the modem planner's press: palm tilted above the target).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import params as P
from .world import Box, World

UP = np.array([0., 0., 1.])
MARKER_OUTSIDE_M = 0.012   # the marker centre lies this far outside the scanner front face (measured on the exported scanner pose)
PRESS_DEPTH_M = P.PRESS_DEPTH_M
PRE_OFFSET_M = P.PRE_OFFSET_M
NEXTTO_RATIO = 1.0 / 6.0
FINGER_SPREAD_M = 0.05    # open gripper: each finger 5 cm from the midpoint
WORLD_X, WORLD_Y = np.array([1., 0., 0.]), np.array([0., 1., 0.])
FACE_ALIGN_COS = math.cos(math.radians(40.))   # the pushed face may be this far from square to the push direction
GAP_GOAL_M = P.GAP_GOAL_M
MIN_GAIN_M = 0.01           # a push must close the gap by at least this much
EDGE_MARGIN_M = P.EDGE_MARGIN_M
FRONT_SPREAD_M = 0.02     # on the button face the fingers are held close, so neither comes near the marker


def unit(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def horiz(v: Sequence[float]) -> np.ndarray:
    v = np.array(v, float)
    v[2] = 0.
    return unit(v)


def rot_yaw(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def tool_rotation(axis: Sequence[float]) -> np.ndarray:
    """Gripper rotation for a tool axis.
    Gripper link orientation (columns x, y, z in the world) whose fingers point along `axis`."""
    z = -unit(axis)
    x = unit(np.cross(UP, z))
    if np.linalg.norm(x) < 1e-6:          # tool vertical: any horizontal x
        x = np.array([1., 0., 0.])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def jaw_rotation(axis: Sequence[float], jaw: Sequence[float]) -> np.ndarray:
    """Gripper rotation for a tool axis and a jaw direction.
    Gripper link orientation whose fingers point along `axis` and whose jaw (finger separation, link y axis) is the horizontal direction `jaw`
    (made orthogonal to the axis). With the gripper open both fingertips touch a face at once, one on each side of the push line."""
    z = -unit(axis)
    y = np.asarray(jaw, float) - float(np.asarray(jaw, float) @ z) * z
    if np.linalg.norm(y) < 1e-6:
        y = np.cross(z, np.array([1., 0., 0.]))
    y = unit(y)
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def tool_axis(direction_h: Sequence[float], tilt: float) -> np.ndarray:
    """Tip direction pitched down.
    Tip direction: horizontal `direction_h`, pitched down by `tilt` rad (0 horizontal, pi/2 straight down)."""
    return unit(math.cos(tilt) * horiz(direction_h) - math.sin(tilt) * UP)


def box_gap(a: Box, b: Box) -> np.ndarray:
    return np.maximum(0., np.maximum(a.lo, b.lo) - np.minimum(a.hi, b.hi))


def nextto_threshold(a: Box, b: Box) -> float:
    """Native NextTo distance bound.
    Native NextTo distance bound: (sum of the six AABB extents of both) / 3 / 6."""
    return float(np.sum(a.hi - a.lo) + np.sum(b.hi - b.lo)) / 3.0 * NEXTTO_RATIO


def box_gap_xy(a: Box, b: Box) -> float:
    return float(np.linalg.norm(box_gap(a, b)[:2]))


# ---- scanner frame ---------------------------------------------------------------------------------------------------------------------
class ScannerFrame:
    """Oriented box of the scanner.
    Oriented box of the scanner on the desk: centre (xy), unit axes, half extents, and the face that carries the button. The root frame +x of
    the scanner points out of the button face, so the front is always +ex."""

    def __init__(self, s: World) -> None:
        sc = s.scanner
        self.ex = np.array([math.cos(sc.yaw), math.sin(sc.yaw), 0.])
        self.ey = np.array([-math.sin(sc.yaw), math.cos(sc.yaw), 0.])
        self.half = np.array(sc.half, float)
        self.centre = np.array([sc.centre[0], sc.centre[1], (sc.z_lo + sc.z_hi) / 2])
        self.z_lo, self.z_hi = float(sc.z_lo), float(sc.z_hi)
        self.front, self.front_axis = self.ex.copy(), 0
        self.marker_lateral = 0.

    def faces(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, float]]:
        """name -> (outward normal, face centre, half width along the face)."""
        out = {}
        for name, axis, sign in (('+x', self.ex, 1.), ('-x', self.ex, -1.), ('+y', self.ey, 1.), ('-y', self.ey, -1.)):
            k = 0 if axis is self.ex else 1
            n = axis * sign
            out[name] = (n, self.centre + n * self.half[k], float(self.half[1 - k]))
        return out

    def face_of(self, normal: np.ndarray) -> str:
        return max(self.faces().items(), key=lambda kv: float(kv[1][0] @ normal))[0]

    def halfspan(self, direction: np.ndarray) -> float:
        """Half extent of the box along a horizontal unit direction."""
        return float(abs(direction @ self.ex) * self.half[0] + abs(direction @ self.ey) * self.half[1])


# ---- button poke -------------------------------------------------------------------------------------------------------------------
def poke_plan(s: World, tilt: float) -> Dict[str, np.ndarray]:
    """Tool path of the button press.
    Tool path that presses the marker: the tip moves along -D (D = outward normal pitched up by tilt) from the pre point to the press point."""
    f = ScannerFrame(s)
    axis = tool_axis(-f.front, tilt)
    d_out = -axis
    return dict(marker=s.marker.copy(), normal=f.front, axis=axis, rotation=tool_rotation(axis), out=d_out,
                pre=s.marker + d_out * PRE_OFFSET_M, press=s.marker + d_out * PRESS_DEPTH_M)


def edge_limit(box: Box, table: Box, e: np.ndarray, margin: float = EDGE_MARGIN_M) -> float:
    """Room to the desk edge along e.
    Largest displacement (m) of `box` along the horizontal unit direction e that keeps its AABB at least `margin` inside the desk top."""
    limit = float('inf')
    for k in (0, 1):
        if e[k] > 1e-9:
            limit = min(limit, (table.hi[k] - margin - box.hi[k]) / e[k])
        elif e[k] < -1e-9:
            limit = min(limit, (table.lo[k] + margin - box.lo[k]) / e[k])
    return max(0., limit)


# ---- push --------------------------------------------------------------------------------------------------------------------------
def push_options(s: World, gap_goal: float = GAP_GOAL_M) -> List[Dict[str, object]]:
    """Pushes towards the laptop, best first.
    Axis-aligned pushes of the scanner towards the laptop, best first. Each: direction e (unit, horizontal, along a scanner axis), the face
    to touch (opposite e), the contact point on it, the travel (m) that brings the AABB gap to GAP_GOAL_M (or, when it cannot get that
    close without leaving the desk, to its smallest value), and the gap left. The button face is touched away from the marker with the
    fingers held close (FRONT_SPREAD_M)."""
    f = ScannerFrame(s)
    lap = s.laptop_box
    v = lap.centre[:2] - f.centre[:2]
    gap0 = float(np.linalg.norm(box_gap(s.scanner_box, lap)))
    out = []
    dirs: List[np.ndarray] = []
    for e in (f.ex, -f.ex, f.ey, -f.ey, WORLD_X, -WORLD_X, WORLD_Y, -WORLD_Y):   # the scanner's own axes and the desk (world) axes
        if not any(float(e @ d) > math.cos(math.radians(3.)) for d in dirs):
            dirs.append(e)
    for e in dirs:
        n_face, _, _ = f.faces()[f.face_of(-e)]
        if float(n_face @ -e) < FACE_ALIGN_COS:
            continue
        limit = min(edge_limit(s.scanner_box, s.table_box, e), 2.5)
        ts = np.arange(0., limit + 1e-9, 0.01)
        gaps = np.array([float(np.linalg.norm(box_gap(Box(s.scanner_box.lo + e * t, s.scanner_box.hi + e * t), lap))) for t in ts])
        hit = np.flatnonzero(gaps <= gap_goal)
        k = int(hit[0]) if len(hit) else int(np.argmin(gaps))
        travel, final = float(ts[k]), float(gaps[k])
        if gap0 - final < MIN_GAIN_M:
            continue
        name = f.face_of(-e)
        n, centre, half_w = f.faces()[name]
        tangent = np.cross(UP, n)
        offset, spread = 0., FINGER_SPREAD_M
        if f.front @ n > 0.9:   # touching the button face: both fingertips stay clear of the marker
            m = float((s.marker - centre) @ tangent)
            offset, spread = -math.copysign(min(0.075, half_w - 0.03), m if abs(m) > 1e-3 else 1.), FRONT_SPREAD_M
        contact = centre + tangent * offset
        contact[2] = f.z_lo + P.PUSH_HEIGHT_M
        out.append(dict(e=e, face=name, normal=n, contact=contact, travel=travel, along=float(v @ e[:2]), final_gap=final, gain=gap0 - final,
                        reaches=bool(final <= gap_goal), spread=spread, offset=offset))
    out.sort(key=lambda o: (0, o['travel']) if o['reaches'] else (1, -o['gain']))
    return out


# ---- base candidates ---------------------------------------------------------------------------------------------------------------
def base_candidates(target_xy: Sequence[float], radii: Sequence[float], bearings: Sequence[float], yaw_offsets: Sequence[float] = (0.,)
                    ) -> List[Tuple[np.ndarray, float]]:
    """Base poses on rings around a target.
    (base xy, base yaw) on rings around the target, facing it (plus yaw offsets). Bearing is the world direction from target to base."""
    out = []
    t = np.asarray(target_xy, float)[:2]
    for r in radii:
        for b in bearings:
            xy = t + r * np.array([math.cos(b), math.sin(b)])
            for dy in yaw_offsets:
                out.append((xy, b + math.pi + dy))
    return out


def to_robot(base: Sequence[float], base_z: float, p_world: Sequence[float]) -> np.ndarray:
    """World point to the base_link frame."""
    c, s = math.cos(base[2]), math.sin(base[2])
    d = np.asarray(p_world, float)
    return np.array([c * (d[0] - base[0]) + s * (d[1] - base[1]), -s * (d[0] - base[0]) + c * (d[1] - base[1]), d[2] - base_z])


def rotation_to_robot(base: Sequence[float], rotation_world: np.ndarray) -> np.ndarray:
    return rot_yaw(-base[2]) @ rotation_world


# ---- NextTo estimate (the native predicate on the perceived boxes) --------------------------------------------------------------------------
HORIZONTAL_RAYS = [np.array([math.cos(a), math.sin(a)]) for a in np.arange(16) * (2 * math.pi / 16)]


def ray_hits_rect(origin: np.ndarray, direction: np.ndarray, centre: np.ndarray, yaw: float, half: Sequence[float]) -> bool:
    """Horizontal ray hits an oriented rectangle.
    Does the horizontal ray origin + t * direction (t > 0) enter the oriented rectangle (centre, yaw, half extents)?"""
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, s], [-s, c]])
    o, d = rot @ (np.asarray(origin, float) - centre), rot @ direction
    t0, t1 = 0., float('inf')
    for k in (0, 1):
        if abs(d[k]) < 1e-12:
            if abs(o[k]) > half[k]:
                return False
        else:
            a, b = (-half[k] - o[k]) / d[k], (half[k] - o[k]) / d[k]
            t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
    return t0 <= t1 and t1 > 0.


def nextto_estimate(w: World, margin: float = 0.0) -> bool:
    """Native NextTo on the perceived boxes.
    The native NextTo(scanner, laptop) on the perceived boxes: the AABB gap norm is at most (sum of the six AABB extents) / 18 minus `margin`,
    and a horizontal ray (16 directions) from the AABB centre of one hits the other (the scanner as its oriented rectangle, the laptop as its box)."""
    a, b = w.scanner_box, w.laptop_box
    if float(np.linalg.norm(box_gap(a, b))) > nextto_threshold(a, b) - margin:
        return False
    sc = w.scanner
    lap_c, lap_half = b.centre[:2], (b.hi[:2] - b.lo[:2]) / 2
    for d in HORIZONTAL_RAYS:
        if ray_hits_rect(a.centre[:2], d, lap_c, 0., lap_half) or ray_hits_rect(b.centre[:2], d, sc.centre, sc.yaw, sc.half):
            return True
    return False
