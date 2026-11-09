"""Pure geometry of the task 55 oracle (numpy only): book frame and faces, desk edges, push and pinch plans, cubby insertion plans, hand
orientations, base candidates.

Conventions: world frame, z up. A tool axis `a` is the unit direction from the wrist to the fingertips; the gripper link z axis is -a, its y axis
is the jaw (the direction in which the two fingers open and close), x completes the frame. For a pinch the tool axis is horizontal and the jaw
vertical (one finger above the book, one below); for a push the tool is nearly vertical with the jaw along the pushed face.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .world import Book, Box, Case, State

UP = np.array([0., 0., 1.])
FINGER_OPEN_M = 0.047          # each finger inner surface this far from the midpoint at command +1 (measured: 0.063 link origin minus 0.0157)
OVERHANG_M = 0.055             # how far the book is pushed beyond the desk edge before the pinch
PINCH_DEPTH_M = 0.035          # fingertips go this far in from the outer edge of the book
MIN_BITE_M = 0.012             # shortest bite that still puts the fingertip on the book and past the desk edge
PUSH_CLEAR_M = 0.03            # the pusher starts this far behind the rear face
EDGES = {'W': np.array([-1., 0., 0.]), 'N': np.array([0., 1., 0.]), 'S': np.array([0., -1., 0.])}   # outward normals of the desk edges used
DESK_MARGIN_M = 0.03           # the pushed book stays this far inside the desk sides it is not pushed towards


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


def jaw_rotation(axis: Sequence[float], jaw: Sequence[float]) -> np.ndarray:
    """Gripper link orientation (columns x, y, z in the world) whose fingers point along `axis` and whose jaw (link y axis) is `jaw` made
    orthogonal to the axis."""
    z = -unit(axis)
    y = np.asarray(jaw, float) - float(np.asarray(jaw, float) @ z) * z
    if np.linalg.norm(y) < 1e-6:
        y = np.cross(z, np.array([1., 0., 0.]))
    y = unit(y)
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def to_robot(base: Sequence[float], base_z: float, p_world: Sequence[float]) -> np.ndarray:
    c, s = math.cos(base[2]), math.sin(base[2])
    d = np.asarray(p_world, float)
    return np.array([c * (d[0] - base[0]) + s * (d[1] - base[1]), -s * (d[0] - base[0]) + c * (d[1] - base[1]), d[2] - base_z])


def to_world(base: Sequence[float], base_z: float, p_robot: Sequence[float]) -> np.ndarray:
    c, s = math.cos(base[2]), math.sin(base[2])
    p = np.asarray(p_robot, float)
    return np.array([base[0] + c * p[0] - s * p[1], base[1] + s * p[0] + c * p[1], base_z + p[2]])


def rotation_to_robot(base: Sequence[float], rotation_world: np.ndarray) -> np.ndarray:
    return rot_yaw(-base[2]) @ rotation_world


def box_gap_xy(a_lo: np.ndarray, a_hi: np.ndarray, b_lo: np.ndarray, b_hi: np.ndarray) -> float:
    d = np.maximum(0., np.maximum(a_lo[:2], b_lo[:2]) - np.minimum(a_hi[:2], b_hi[:2]))
    return float(np.linalg.norm(d))


def boxes_overlap_xy(a_lo, a_hi, b_lo, b_hi, margin: float = 0.) -> bool:
    return bool(np.all(np.minimum(a_hi[:2], b_hi[:2]) - np.maximum(a_lo[:2], b_lo[:2]) > -margin))


# ---- book frame --------------------------------------------------------------------------------------------------------------------------
class BookFrame:
    """Oriented box of a flat book: centre (xy and mid height), unit axes ex, ey (horizontal, along the root-link axes), half extents, thickness."""

    def __init__(self, b: Book) -> None:
        yaw = b.yaw
        self.ex = np.array([math.cos(yaw), math.sin(yaw), 0.])
        self.ey = np.array([-math.sin(yaw), math.cos(yaw), 0.])
        lo, hi = b.local.lo, b.local.hi
        self.half = np.array([(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2])
        centre_local = (lo + hi) / 2
        self.centre = b.pos + centre_local[0] * self.ex + centre_local[1] * self.ey
        self.centre[2] = b.box.centre[2]
        self.z_lo, self.z_hi = float(b.box.lo[2]), float(b.box.hi[2])
        self.thick = self.z_hi - self.z_lo
        if self.half[0] < 0 or self.half[1] < 0:
            raise ValueError('negative extents')
        # a book that lies on its side (thick local z not vertical) is not handled: the extents are taken from the world AABB height

    @property
    def z_mid(self) -> float:
        return 0.5 * (self.z_lo + self.z_hi)

    def corners(self) -> np.ndarray:
        out = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                out.append(self.centre[:2] + sx * self.half[0] * self.ex[:2] + sy * self.half[1] * self.ey[:2])
        return np.array(out)

    def faces(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, float]]:
        """name -> (outward normal, face centre (xy, mid height), half width along the face)."""
        out = {}
        for name, axis, sign, k in (('+x', self.ex, 1., 0), ('-x', self.ex, -1., 0), ('+y', self.ey, 1., 1), ('-y', self.ey, -1., 1)):
            n = axis * sign
            c = self.centre + n * self.half[k]
            out[name] = (n, c, float(self.half[1 - k]))
        return out

    def face_towards(self, direction: Sequence[float]) -> Tuple[str, np.ndarray, np.ndarray, float]:
        """The face whose outward normal is closest to the horizontal unit `direction`: (name, normal, centre, half width)."""
        d = horiz(direction)
        name, (n, c, hw) = max(self.faces().items(), key=lambda kv: float(kv[1][0] @ d))
        return name, n, c, hw

    def extent(self, direction: Sequence[float]) -> float:
        """Largest value of dot(point, direction) over the book footprint (point = xy of a corner)."""
        d = horiz(direction)[:2]
        return float(np.max(self.corners() @ d))

    def span(self, direction: Sequence[float]) -> float:
        """Half extent of the footprint along a horizontal unit direction."""
        d = horiz(direction)
        return float(abs(d @ self.ex) * self.half[0] + abs(d @ self.ey) * self.half[1])


def tilt_between(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = horiz(a), horiz(b)
    return math.acos(float(np.clip(a @ b, -1., 1.)))


# ---- desk edges and pushes ---------------------------------------------------------------------------------------------------------------
def edge_coordinate(table: Box, normal: np.ndarray) -> float:
    """Value of dot(p, normal) on the desk edge whose outward normal is `normal`."""
    return float(table.hi[0]) * 1. if normal[0] > 0.5 else (-float(table.lo[0]) if normal[0] < -0.5 else
                                                             (float(table.hi[1]) if normal[1] > 0.5 else -float(table.lo[1])))


def push_travel(f: BookFrame, table: Box, normal: np.ndarray, overhang: float = OVERHANG_M) -> float:
    """Distance the book must move along `normal` so the corner overhangs by `overhang` and the pinch point is past the edge.

    A skewed book can hang a corner over the edge while the grasp (PINCH_DEPTH_M in from the outer face) is still above the desk.
    Pushing only to the corner then leaves no legal pinch and no further push."""
    n = horiz(normal)
    edge = edge_coordinate(table, n)
    corner = edge + overhang - f.extent(n)
    grasp = np.asarray(pinch_plan(f, n)['grasp'], float)
    clear = edge + 0.012 - float(grasp[:2] @ n[:2])
    return max(corner, clear)


def book_hand_envelope(f: BookFrame, table: Box, normal: np.ndarray, travel: float, reach_out: float = 0.30, half_width: float = 0.09,
                       z_lo: float = 0.60, z_hi: float = 1.0, lateral: float = 0.) -> Box:
    """World box outside the desk edge where the hand and fingers sit during the pinch of the book after it is pushed `travel` along `normal`:
    from the edge out to `reach_out`, `half_width` either side of the book centre across the edge, z between the given heights."""
    n = horiz(normal)
    m = np.array([-n[1], n[0], 0.])
    c = f.centre + n * travel + m * lateral
    edge = edge_coordinate(table, n)
    start = n * edge
    pts = []
    for along in (0., reach_out):
        for across in (-half_width, half_width):
            # points: on the edge line (coordinate edge along n) and `reach_out` beyond, at the book's lateral position
            p = n[:2] * (edge + along) + m[:2] * float(c[:2] @ m[:2] + across)
            pts.append(p)
    pts = np.array(pts)
    return Box(np.array([pts[:, 0].min(), pts[:, 1].min(), z_lo]), np.array([pts[:, 0].max(), pts[:, 1].max(), z_hi]))


Rect = Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]   # centre xy, unit axes ex, ey (xy), half extents


def rect_of(f: 'BookFrame') -> Rect:
    return (f.centre[:2].copy(), f.ex[:2].copy(), f.ey[:2].copy(), float(f.half[0]), float(f.half[1]))


def rect_of_box(lo: Sequence[float], hi: Sequence[float]) -> Rect:
    lo, hi = np.asarray(lo, float)[:2], np.asarray(hi, float)[:2]
    return ((lo + hi) / 2, np.array([1., 0.]), np.array([0., 1.]), float((hi[0] - lo[0]) / 2), float((hi[1] - lo[1]) / 2))


def rects_overlap(a: Rect, b: Rect, pad: float = 0.) -> bool:
    """Separating axis test of two oriented rectangles in the plane; pad grows both."""
    for r in (a, b):
        for axis in (r[1], r[2]):
            ca, cb = float(a[0] @ axis), float(b[0] @ axis)
            ra = a[3] * abs(float(a[1] @ axis)) + a[4] * abs(float(a[2] @ axis))
            rb = b[3] * abs(float(b[1] @ axis)) + b[4] * abs(float(b[2] @ axis))
            if abs(ca - cb) > ra + rb + 2 * pad:
                return False
    return True


def sweep_hits(f: 'BookFrame', normal: np.ndarray, travel: float, others: Sequence[Rect], pad: float = 0.004, step: float = 0.02) -> Optional[Rect]:
    """First obstacle rectangle that the book rectangle meets while moved along `normal` by 0..travel, or None."""
    n = horiz(normal)[:2]
    c0, ex, ey, hx, hy = rect_of(f)
    for t in list(np.arange(0., travel, step)) + [travel]:
        moving = (c0 + n * t, ex, ey, hx, hy)
        for o in others:
            if rects_overlap(moving, o, pad):
                if t == 0.:
                    continue       # already touching at the start (neighbours on the desk): only a collision that develops counts
                return o
    return None


def swept_boxes(f: BookFrame, normal: np.ndarray, travel: float, pad: float = 0.005, start: float = 0.03, step: float = 0.03) -> List[Box]:
    """Footprint boxes (AABB of the corners, padded) of the book moved along `normal` by distances from `start` to `travel`: the start footprint is
    left out, a book that already touches its neighbour may leave it."""
    n = horiz(normal)[:2]
    out = []
    ts = list(np.arange(start, travel, step)) + [travel]
    for t in ts:
        pts = f.corners() + n * t
        out.append(Box(np.array([pts[:, 0].min() - pad, pts[:, 1].min() - pad, f.z_lo - 0.01]),
                       np.array([pts[:, 0].max() + pad, pts[:, 1].max() + pad, f.z_hi + 0.01])))
    return out


def lateral_room(f: BookFrame, table: Box, normal: np.ndarray) -> Tuple[float, float]:
    """How far (m) the book can slide along the edge, (negative direction, positive direction of m = up x normal ... rotated normal), inside the desk."""
    n = horiz(normal)
    m = np.array([-n[1], n[0], 0.])
    cm = float(f.centre[:2] @ m[:2])
    sp = f.span(m)
    corners = np.array([[table.lo[0], table.lo[1]], [table.hi[0], table.lo[1]], [table.lo[0], table.hi[1]], [table.hi[0], table.hi[1]]])
    proj = corners @ m[:2]
    lo, hi = float(proj.min()) + DESK_MARGIN_M, float(proj.max()) - DESK_MARGIN_M
    return (cm - sp) - lo, hi - (cm + sp)


def pusher_pose(f: BookFrame, direction: np.ndarray, z_tip: float, lateral: float = 0., clear: float = PUSH_CLEAR_M) -> Dict[str, np.ndarray]:
    """Where the pusher (two fingertips spread along the rear face, tool vertical) stands before pushing the book along `direction`:
    `lateral` along the face from its centre, `clear` behind it, fingertip height z_tip. Returns the contact start point and the jaw direction."""
    d = horiz(direction)
    name, n, c, hw = f.face_towards(-d)        # the rear face
    jaw = np.cross(UP, n)
    start = c + n * clear + jaw * lateral
    start = np.array([start[0], start[1], z_tip])
    return dict(start=start, jaw=jaw, push=d, face=name, rear_normal=n, half_width=hw)


# ---- pinch ---------------------------------------------------------------------------------------------------------------------------------
def pinch_plan(f: BookFrame, normal: np.ndarray, out_clear: float = 0.12) -> Dict[str, object]:
    """Pinch of the overhanging face that looks along `normal` (outside the desk): tool axis -u (into the book), jaw vertical, finger midpoint
    PINCH_DEPTH_M in from the outer face at mid height. pre = the same point `out_clear` further out."""
    name, u, c, hw = f.face_towards(normal)
    mid = np.array([c[0] - u[0] * PINCH_DEPTH_M, c[1] - u[1] * PINCH_DEPTH_M, f.z_mid])
    axis = -u
    return dict(face=name, normal=u, axis=axis, rotation=jaw_rotation(axis, UP), grasp=mid, pre=mid + u * out_clear, half_width=hw,
                skew=tilt_between(u, normal))


def pinch_overhang(f: BookFrame, table: Box, normal: np.ndarray, margin: float = 0.003, min_bite: float = MIN_BITE_M,
                   out_clear: float = 0.12) -> List[Dict[str, object]]:
    """Pinch poses whose fingertip point is past the desk edge, deepest bite first.

    `pinch_plan` bites PINCH_DEPTH_M in from the middle of the outer face. A rotated book can hang one corner over the edge while that
    midpoint is still above the desk, and a corner that only just overhangs cannot take a full 35 mm bite. Slide along the face toward the
    corner that sticks out, and shorten the bite so the fingertip stays past the edge. A square pinch (approach along the face) is kept
    when the face is within 30 deg of the edge and that midpoint is already past the desk; otherwise the hand comes straight in from
    outside the desk, along the edge normal, so a skewed face does not send the wrist on a diagonal over the desktop. Empty when no
    point on the face gives a bite of at least `min_bite` past the edge."""
    name, u, centre, hw = f.face_towards(normal)
    n = horiz(normal)
    edge = edge_coordinate(table, n)
    un = float(u[:2] @ n[:2])
    if un < 0.2:
        return []
    tangent = np.array([-u[1], u[0], 0.])
    if float(tangent[:2] @ n[:2]) < 0.:
        tangent = -tangent
    skew = tilt_between(u, n)
    base = pinch_plan(f, n, out_clear=out_clear)
    limit = max(0., float(hw) - 0.008)
    stations = [0.]
    if limit > 0.015:
        stations += [float(s) for s in np.linspace(0.02, limit, 6)]
        stations += [-float(s) for s in np.linspace(0.02, limit, 6)]
    found = []
    for s in stations:
        face = np.array([centre[0], centre[1], f.z_mid]) + tangent * s
        room = (float(face[:2] @ n[:2]) - edge - margin) / un
        if room < min_bite:
            continue
        depth = min(PINCH_DEPTH_M, float(room))
        grasp = np.array([face[0] - u[0] * depth, face[1] - u[1] * depth, f.z_mid])
        clearance = float(grasp[:2] @ n[:2]) - edge
        if clearance < margin - 1e-9:
            continue
        # a grasp that only just clears the edge puts the finger pad back on the desktop; prefer one that is clearly past it
        found.append((depth, 1. if clearance >= 0.012 else 0., -abs(s), s, grasp))
    found.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    square: List[Dict[str, object]] = []
    # the first book was lifted by this square pinch (face within 30 deg, fingertips at or past the edge)
    centre_past = float(np.asarray(base['grasp'])[:2] @ n[:2]) >= edge - 0.002
    if centre_past and skew <= math.radians(30):
        hist = dict(base)
        hist['depth'] = PINCH_DEPTH_M
        hist['along'] = 0.
        square.append(hist)
    straight: List[Dict[str, object]] = []
    axis = -n
    for depth, _, _, s, grasp in found:
        # a near-aligned full bite is already the square plan; the other stations still dodge a blocked hand box
        if centre_past and skew <= math.radians(10) and abs(s) < 1e-9:
            continue
        straight.append(dict(face=name, normal=n, axis=axis, rotation=jaw_rotation(axis, UP), grasp=grasp,
                             pre=grasp + n * out_clear, half_width=hw, skew=skew, depth=float(depth), along=float(s)))
    return square + straight


# ---- cubbies and insertion ---------------------------------------------------------------------------------------------------------------
def top_cubbies(cases: Dict[str, Case]) -> List[Tuple[str, int, Box]]:
    """(bookcase name, index, cubby box) of the top-row cubbies of every bookcase (the cubbies whose top is the highest)."""
    out = []
    for name, case in cases.items():
        top = max(float(c.hi[2]) for c in case.cubbies)
        for i, c in enumerate(case.cubbies):
            if float(c.hi[2]) > top - 0.05:
                out.append((name, i, c))
    return out


def insert_plan(cubby: Box, case_box: Box, book_size_along_axis: float, tip_to_centre: float, thick: float, shelf_clear: float = 0.025,
                depth_frac: float = 0.45) -> Dict[str, object]:
    """Insertion of a flat book carried with the tool axis +y (the opening of the bookcase faces -y). The book centre goes to `depth_frac` of the
    cubby depth from its front, centred across, its bottom `shelf_clear` above the shelf (the cubby floor). tip_to_centre = distance from the
    finger midpoint to the book centre along the tool axis (measured at the grasp). Returns the target of the book centre and of the finger
    midpoint, and the pre point 0.15 m in front of the opening."""
    front = float(cubby.lo[1])
    depth = float(cubby.hi[1]) - front
    cx = float((cubby.lo[0] + cubby.hi[0]) / 2)
    centre_y = front + depth_frac * depth
    centre_z = float(cubby.lo[2]) + shelf_clear + thick / 2
    centre = np.array([cx, centre_y, centre_z])
    axis = np.array([0., 1., 0.])
    tip = centre - axis * tip_to_centre
    return dict(axis=axis, rotation=jaw_rotation(axis, UP), centre=centre, tip=tip, pre=np.array([cx, float(tip[1]) - 0.17, centre_z]),
                front_y=front, depth=depth, rotation_jaw=UP)


# ---- base candidates ---------------------------------------------------------------------------------------------------------------------
def base_candidates(target_xy: Sequence[float], radii: Sequence[float], bearings: Sequence[float], yaw_offsets: Sequence[float] = (0.,)
                    ) -> List[Tuple[np.ndarray, float]]:
    """(base xy, base yaw) on rings around the target, facing it (plus yaw offsets). Bearing is the world direction from target to base."""
    out = []
    t = np.asarray(target_xy, float)[:2]
    for r in radii:
        for b in bearings:
            xy = t + r * np.array([math.cos(b), math.sin(b)])
            for dy in yaw_offsets:
                out.append((xy, b + math.pi + dy))
    return out
