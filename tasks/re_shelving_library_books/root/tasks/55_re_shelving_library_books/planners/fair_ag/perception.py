"""Perception of the fair planner of task 55: the native observation (3 RGB-D cameras + proprio, camera calibration) to the State the logic plans
on, in the frame of the offline scene map.

  pose     odometry from the measured base velocity, an odometry -> map transform from the localisation (localise.py) after a turn in place,
           kept up by matching every head view to the map (the base velocity alone drifts by tens of percent on slow turns), and corrected
           at the desk by the desk top edges seen in the head cloud
  books    points 0.6-6 cm above the desk top (head and wrist clouds) that are not under a tall desk item, clustered on a 1 cm grid; every
           cluster is fitted by a rectangle and matched to the three book models by size; a book in the hand rides with the gripper frame
  desk items  points 7-60 cm above the desk top: the lamp footprint (an obstacle of the hand)
  inside   a book is believed inside when its estimate (riding in the hand) has its centre in a cubby of the map when the hand opens
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from core.skills_lib import r1pro_kinematics as K
from core.skills_lib import rgbd_geometry as RG

from . import cloud as C
from . import localise as L
from .scene_map import scene_map
from .world import BOOKS, Book, Box, State, yaw_quat

LOW_BAND = (0.006, 0.060)       # above the visual desk top: book top surfaces
TALL_BAND = (0.075, 0.60)       # desk items (the lamp)
DESK_GROW_M = 0.14              # books may overhang the desk this far
GRID_M = 0.01
BOOK_MIN_CELLS = 60
MATCH_MAX = 0.06                # summed extent error (m) of a cluster against a book model
TRACK_SPAN = (0.05, math.radians(1.5))     # local refine window per head view (m, rad)
TRACK_GAIN, TRACK_MAX_STEP = 0.5, (0.02, math.radians(1.0))
TRACK_MIN_POINTS = 120
DESK_EDGE_BAND_M = 0.06
DESK_CORR_MAX_M = 0.08
FINGER_HELD_MIN, FINGER_HELD_MAX = 0.006, 0.040   # mean finger joint (m) of a closed gripper that holds something
INSIDE_FRONT_MARGIN_M = 0.05    # the book centre is this far behind the cubby front before it counts as inside
HAND_CLEAR_M = 0.16             # points this near a finger midpoint (or a book riding in the hand) are not desk objects
EDGE_VIEW_MIN = 0.7             # share of the samples just past a desk edge that must project into the head image for that edge to be used


def rot2(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


class Perception:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.map = scene_map()
        self.odo = RG.Odometry()
        self.T: Optional[np.ndarray] = None
        self.hyps: List[Dict[str, Any]] = []
        self.scan_pts: List[np.ndarray] = []
        self.books: Dict[str, Book] = {}
        self.seen_step: Dict[str, int] = {}
        self.items: List[Tuple[np.ndarray, np.ndarray]] = []
        self.attached: Optional[Dict[str, Any]] = None
        self.inside: Dict[str, bool] = {}
        self.frozen: set = set()
        self.arrays: Dict[str, np.ndarray] = {}
        self.q: Optional[np.ndarray] = None
        self.step = 0
        self.desk_corr = np.zeros(2)
        self.counts = dict(track=0, track_moves=0, desk_fix=0, book_updates=0, merged=0)
        self.last_track: Dict[str, float] = {}

    # ---- every control ------------------------------------------------------------------------------------------------------------------
    def observe(self, obs: Dict[str, Any], step: int) -> np.ndarray:
        self.arrays = C.obs_arrays(obs)
        self.step = step
        self.odo.update(self.arrays['proprio'][:3])
        self.q = K.q_from_proprio(self.arrays['proprio'])
        return self.q

    def base(self) -> np.ndarray:
        if self.T is None:
            return self.odo.pose.copy()
        return L.compose(self.T, self.odo.pose)

    def head_robot(self, stride: int = 4) -> np.ndarray:
        return C.camera_points(self.arrays['head'], 'head', self.arrays['cam_rel'], self.q, stride)

    # ---- localisation ---------------------------------------------------------------------------------------------------------------------
    def collect_scan(self) -> None:
        p = self.head_robot(4)
        z = p[:, 2] + self.map.base_z
        p = p[(z > L.BAND[0]) & (z < L.BAND[1])]
        self.scan_pts.append(C.to_map(p, self.odo.pose, 0.)[:, :2])

    def localise(self) -> bool:
        pts = np.concatenate(self.scan_pts) if self.scan_pts else np.zeros((0, 2))
        self.hyps = L.global_search(self.map, pts, self.odo.pose)
        if not self.hyps:
            return False
        self.T = np.asarray(self.hyps[0]['T'], float)
        return True

    def track(self) -> None:
        """Refine the transform with the current head view (small window), moving it part of the way and by a capped step."""
        if self.T is None:
            return
        p = self.head_robot(4)
        z = p[:, 2] + self.map.base_z
        p = p[(z > L.BAND[0]) & (z < L.BAND[1])]
        pts = L.unique_cells(C.to_map(p, self.odo.pose, 0.)[:, :2])
        if len(pts) < TRACK_MIN_POINTS:
            return
        self.counts['track'] += 1
        s0 = L.score(self.map, pts, self.T)
        T, s1 = L.refine(self.map, pts, self.T, TRACK_SPAN[0], TRACK_SPAN[1], steps=2)
        self.last_track = dict(before=round(s0, 3), after=round(s1, 3), n=len(pts))
        if s1 <= s0 + 0.005:
            return
        d = (T - self.T) * TRACK_GAIN
        d[:2] = np.clip(d[:2], -TRACK_MAX_STEP[0], TRACK_MAX_STEP[0])
        d[2] = float(np.clip(d[2], -TRACK_MAX_STEP[1], TRACK_MAX_STEP[1]))
        self.T = self.T + d
        self.counts['track_moves'] += 1

    def apply_map_shift(self, dxy: np.ndarray) -> None:
        self.T = self.T + np.r_[dxy, 0.]

    def desk_fix(self) -> Optional[np.ndarray]:
        """Shift the transform so the desk top edges seen in the head cloud sit on the map's desk edges (only the edges with enough points)."""
        if self.T is None:
            return None
        m, t = self.map, self.map.table
        w = C.to_map(self.head_robot(2), self.base(), m.base_z)
        top = w[(np.abs(w[:, 2] - m.desk_top) < 0.010)]
        top = top[np.all(top[:, :2] > t.lo[:2] - 0.08, 1) & np.all(top[:, :2] < t.hi[:2] + 0.08, 1)]
        if len(top) < 400:
            return None
        corr = np.zeros(2)
        votes = np.zeros(2)
        base = self.base()
        for axis in (0, 1):
            other = 1 - axis
            inner = (top[:, other] > t.lo[other] + 0.05) & (top[:, other] < t.hi[other] - 0.05)
            for sign, edge in ((-1., m.desk_edges[axis][0]), (1., m.desk_edges[axis][1])):
                near = top[inner & (np.abs(top[:, axis] - edge) < DESK_EDGE_BAND_M)]
                if len(near) < 40 or not self.edge_in_view(axis, sign, edge, base):
                    continue
                seen = float(np.percentile(sign * near[:, axis], 99.0)) * sign
                corr[axis] += edge - seen
                votes[axis] += 1
        if not votes.any():
            return None
        corr = np.where(votes > 0, corr / np.maximum(votes, 1), 0.)
        self.desk_votes = votes.tolist()
        if np.any(np.abs(corr) > DESK_CORR_MAX_M):
            return None
        self.apply_map_shift(corr)
        self.desk_corr = corr
        self.counts['desk_fix'] += 1
        return corr

    def edge_in_view(self, axis: int, sign: float, edge: float, base: np.ndarray) -> bool:
        """Samples 4 cm past a desk edge (at the top height, along its inner part) project into the head image with a margin: the end of
        the desk-top points there is the edge and not the border of the view."""
        t, m = self.map.table, self.map
        other = 1 - axis
        s = np.linspace(t.lo[other] + 0.06, t.hi[other] - 0.06, 12)
        pts = np.zeros((len(s), 3))
        pts[:, axis] = edge + sign * 0.04
        pts[:, other] = s
        pts[:, 2] = m.desk_top
        c, sn = math.cos(base[2]), math.sin(base[2])
        d = pts[:, :2] - base[:2]
        rob = np.stack([c * d[:, 0] + sn * d[:, 1], -sn * d[:, 0] + c * d[:, 1], pts[:, 2] - m.base_z], 1)
        px, depth = RG.project(rob, 'head', self.arrays['cam_rel'])
        size = RG.CAM_SIZE_FX['head'][0]
        ok = (depth > 0.2) & np.all(px > 15, 1) & np.all(px < size - 15, 1)
        return float(ok.mean()) >= EDGE_VIEW_MIN

    # ---- desk objects ---------------------------------------------------------------------------------------------------------------------
    def desk_points(self) -> np.ndarray:
        base, m = self.base(), self.map
        parts = [C.to_map(self.head_robot(2), base, m.base_z)]
        for cam in ('left_wrist', 'right_wrist'):
            parts.append(C.to_map(C.camera_points(self.arrays[cam], cam, self.arrays['cam_rel'], self.q, 3, max_range=1.5), base, m.base_z))
        w = np.concatenate(parts)
        t = m.table
        return w[np.all(w[:, :2] > t.lo[:2] - DESK_GROW_M, 1) & np.all(w[:, :2] < t.hi[:2] + DESK_GROW_M, 1)]

    def update_desk(self) -> List[str]:
        """Detect the books and desk items in the current views; returns the names of the books updated."""
        if self.T is None:
            return []
        m = self.map
        w = self.desk_points()
        for side in ('left', 'right'):              # the hands and a book riding in one are not desk objects
            w = w[np.linalg.norm(w - self.hand(side)[0], axis=1) > HAND_CLEAR_M]
        if self.attached is not None:
            w = w[np.linalg.norm(w - self.books[self.attached['book']].box.centre, axis=1) > HAND_CLEAR_M]
        dz = w[:, 2] - m.desk_top
        tall = w[(dz > TALL_BAND[0]) & (dz < TALL_BAND[1])]
        low = w[(dz > LOW_BAND[0]) & (dz < LOW_BAND[1])]
        if len(low) < BOOK_MIN_CELLS:
            return []
        both = np.concatenate([low[:, :2], tall[:, :2]])
        lo = both.min(0) - 0.1
        shape = tuple(((both.max(0) + 0.1 - lo) / GRID_M).astype(int) + 1)
        g_low = np.zeros(shape, bool)
        il = ((low[:, :2] - lo) / GRID_M).astype(int)
        g_low[il[:, 0], il[:, 1]] = True
        items = []
        if len(tall):
            g_tall = np.zeros(shape, bool)
            it = ((tall[:, :2] - lo) / GRID_M).astype(int)
            g_tall[it[:, 0], it[:, 1]] = True
            g_tall = ndimage.binary_opening(g_tall, iterations=1)
            for _, idx in RG.clusters(g_tall, 15):
                a, b = lo + idx.min(0) * GRID_M, lo + (idx.max(0) + 1) * GRID_M
                items.append((a, b))
            g_low &= ~ndimage.binary_dilation(g_tall, iterations=5)
        self.items = items
        g_low = ndimage.binary_closing(g_low, iterations=1)
        clusters = []
        for _, idx in RG.clusters(g_low, BOOK_MIN_CELLS):
            xy = lo + (idx + 0.5) * GRID_M
            centre, a, ext = RG.min_area_rect(xy)
            sel = low[np.linalg.norm(low[:, :2] - centre, axis=1) < 0.5 * float(np.max(ext))]
            top = float(np.percentile(sel[:, 2], 90)) if len(sel) else m.desk_top + 0.03
            clusters.append(dict(centre=centre, yaw=a, ext=ext, top=top, cells=len(idx)))
        return self._assign(clusters)

    def _assign(self, clusters: List[Dict[str, Any]]) -> List[str]:
        m = self.map
        names = [b for b in BOOKS if b not in self.frozen and not self.inside.get(b) and not (self.attached and self.attached['book'] == b)]
        cost = {}
        for i, c in enumerate(clusters):
            for b in names:
                mb = m.books[b]
                dims = np.sort((mb.hi - mb.lo)[:2])
                cost[(i, b)] = float(np.sum(np.abs(np.sort(c['ext']) - dims)))
        used_c, done = set(), []
        for (i, b), v in sorted(cost.items(), key=lambda kv: kv[1]):
            if v > MATCH_MAX or i in used_c or b in done:
                continue
            used_c.add(i)
            done.append(b)
            self.books[b] = self._book(b, clusters[i])
            self.seen_step[b] = self.step
        self.counts['book_updates'] += len(done)
        self.counts['merged'] += sum(1 for i, c in enumerate(clusters) if i not in used_c and np.prod(c['ext']) > 0.04)
        return done

    def _book(self, name: str, c: Dict[str, Any]) -> Book:
        mb = self.map.books[name]
        dims = (mb.hi - mb.lo)
        yaw = c['yaw'] if (c['ext'][0] > c['ext'][1]) == (dims[0] > dims[1]) else c['yaw'] + math.pi / 2
        half = dims / 2
        local = Box(-half, half)
        z_hi = max(c['top'], self.map.desk_top + dims[2] * 0.6)
        z_lo = z_hi - dims[2]
        pos = np.r_[c['centre'], (z_lo + z_hi) / 2]
        return Book(pos, yaw_quat(yaw), self._aabb(pos, yaw, half), local)

    @staticmethod
    def _aabb(pos: np.ndarray, yaw: float, half: np.ndarray) -> Box:
        r = rot2(yaw)
        corners = np.array([r @ np.array([sx * half[0], sy * half[1]]) for sx in (-1, 1) for sy in (-1, 1)]) + pos[:2]
        return Box(np.r_[corners.min(0), pos[2] - half[2]], np.r_[corners.max(0), pos[2] + half[2]])

    # ---- a book in the hand ---------------------------------------------------------------------------------------------------------------
    def hand(self, side: str) -> Tuple[np.ndarray, np.ndarray]:
        """Finger midpoint and gripper rotation of a hand in the map frame."""
        servo = K.ArmServo(side)
        servo.use_fingers_mid()
        pose = servo.pose(self.q)
        b = self.base()
        r = np.eye(3)
        r[:2, :2] = rot2(b[2])
        return C.to_map(pose[:3, 3][None], b, self.map.base_z)[0], r @ pose[:3, :3]

    def finger_gap(self, side: str) -> float:
        return float(np.mean(self.q[list(K.FINGER_DOFS[side])]))

    def holds(self, side: str) -> bool:
        """The fingers of a closed gripper stopped on something (an empty one closes to about 0, an open one stands at 0.05)."""
        return FINGER_HELD_MIN < self.finger_gap(side) < FINGER_HELD_MAX

    def attach(self, book: str, side: str) -> None:
        """The book now rides with the gripper: its pose in the gripper frame is fixed from the last estimate."""
        if book not in self.books:
            return
        fm, r = self.hand(side)
        b = self.books[book]
        self.attached = dict(book=book, side=side, off=r.T @ (b.pos - fm), dyaw=b.yaw - math.atan2(r[1, 0], r[0, 0]))

    def detach(self, cubby_check: bool = True) -> Optional[str]:
        if self.attached is None:
            return None
        book = self.attached['book']
        self._ride()
        self.attached = None
        if cubby_check and self.in_cubby(book):
            self.inside[book] = True
        return book

    def _ride(self) -> None:
        a = self.attached
        fm, r = self.hand(a['side'])
        pos = fm + r @ a['off']
        yaw = math.atan2(r[1, 0], r[0, 0]) + a['dyaw']
        b = self.books[a['book']]
        half = (b.local.hi - b.local.lo) / 2
        self.books[a['book']] = Book(pos, yaw_quat(yaw), self._aabb(pos, yaw, half), b.local)

    def in_cubby(self, book: str) -> bool:
        b = self.books.get(book)
        if b is None:
            return False
        c = b.box.centre
        for case in self.map.cases.values():
            for cub in case.cubbies:
                lo = cub.lo + np.array([0.005, INSIDE_FRONT_MARGIN_M, 0.])
                if np.all(c >= lo) and np.all(c <= cub.hi):
                    return True
        return False

    # ---- the State for the logic -------------------------------------------------------------------------------------------------------------
    def state(self, step: int) -> State:
        if self.attached is not None:
            self._ride()
        m = self.map
        atoms = {f'inside_{b}': bool(self.inside.get(b)) for b in BOOKS}
        if self.attached is not None and self.in_cubby(self.attached['book']):
            atoms[f"inside_{self.attached['book']}"] = True
        t = m.table
        obstacles = [(t.lo[:2].tolist(), t.hi[:2].tolist(), 'desk')]
        obstacles += [(c.box.lo[:2].tolist(), c.box.hi[:2].tolist(), 'bookcase') for c in m.cases.values()]
        obstacles += [(a.tolist(), b.tolist(), 'table_lamp') for a, b in self.items]
        obstacles += self.clutter
        return State(step=step, base=self.base(), base_z=m.base_z, books=dict(self.books), table_box=t, cases=m.cases, atoms=atoms,
                     obstacles=obstacles)

    @property
    def clutter(self) -> List[tuple]:
        if not hasattr(self, '_clutter'):
            m = self.map
            grow = np.array([0.03, 0.03, 0.])
            skip = [Box(m.table.lo - grow, m.table.hi + grow)] + [Box(c.box.lo - grow, c.box.hi + grow) for c in m.cases.values()]
            self._clutter = m.clutter_boxes(m.table, 0.9, skip)
        return self._clutter

    def sheet_points(self, xy: np.ndarray, radius: float = 0.45) -> np.ndarray:
        """Book-top points near `xy` (map frame) from the current cameras.

        The desk top itself is below the band. Points within 3.5 cm of a finger midpoint are the fingers; the rest of a
        book pressed against the paddle stays, unlike the 0.16 m mask used when fitting models. Empty before localisation."""
        if self.T is None or self.q is None or not self.arrays:
            return np.zeros((0, 3))
        base, m = self.base(), self.map
        parts = []
        head = C.to_map(self.head_robot(2), base, m.base_z)
        if len(head):
            parts.append(head)
        for cam in ('left_wrist', 'right_wrist'):
            pts = C.camera_points(self.arrays[cam], cam, self.arrays['cam_rel'], self.q, 2, max_range=1.6)
            pts = C.to_map(pts, base, m.base_z)
            if len(pts):
                parts.append(pts)
        if not parts:
            return np.zeros((0, 3))
        w = np.concatenate(parts)
        dz = w[:, 2] - m.desk_top
        w = w[(dz > 0.008) & (dz < 0.055)]
        if len(w) == 0:
            return w
        w = w[np.linalg.norm(w[:, :2] - np.asarray(xy, float)[:2], axis=1) < radius]
        for side in ('left', 'right'):
            if len(w) == 0:
                break
            w = w[np.linalg.norm(w - self.hand(side)[0], axis=1) > 0.035]
        return w

    def est(self) -> Dict[str, Any]:
        b = self.base()
        out: Dict[str, Any] = dict(base_map=[round(float(v), 3) for v in b], T=None if self.T is None else [round(float(v), 4) for v in self.T],
                                   hyps=[dict(T=[round(float(v), 3) for v in h['T']], score=round(float(h['score']), 3)) for h in self.hyps[:4]],
                                   books_map={k: [round(float(v), 3) for v in bk.box.centre] + [round(bk.yaw, 3)] for k, bk in self.books.items()},
                                   items=[[round(float(v), 3) for v in np.r_[a, bb]] for a, bb in self.items], inside=dict(self.inside),
                                   attached=None if self.attached is None else self.attached['book'], desk_corr=[round(float(v), 4) for v in self.desk_corr],
                                   track=self.last_track, counts=dict(self.counts))
        return out
