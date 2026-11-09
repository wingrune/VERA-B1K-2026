"""The cabinet for the fair planner of task 83: its top is a horizontal surface at CAB_TOP_BAND_M of the prior size (CAB_LENGTH_M x
CAB_DEPTH_M). A decaying layer of the top-band points of the last frames (only cells with nothing seen above them: walls and door frames have
points at every height) is clustered; a cluster whose rectangle along the room axes has the prior depth, enough length and is filled is the
cabinet. The front is the long side that faces the robot (the back stands against a wall); the drawer face is measured as the vertical surface
just in front of that edge. The ledges of the top drawer follow from the asset priors (world.CabinetEst.ledge)."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

from core.skills_lib import rgbd_geometry as G

from .. import params as P
from ..world import CabinetEst


class CabinetMixin:
    def _top_cells(self, pts: np.ndarray) -> np.ndarray:
        c = np.floor((pts[:, :2] + P.MAP_HALF_M) / P.TOP_RES_M).astype(int)
        n = self.top.shape[0]
        return np.unique(c[(c >= 0).all(1) & (c < n).all(1)], axis=0)

    def _update_top(self, pts_odom: np.ndarray) -> None:
        self.top *= P.TOP_DECAY
        z = pts_odom[:, 2]
        band = self._top_cells(pts_odom[(z > P.CAB_TOP_BAND_M[0]) & (z < P.CAB_TOP_BAND_M[1])])
        if len(band):
            above = np.zeros(self.top.shape, bool)
            c = self._top_cells(pts_odom[(z > P.CAB_TOP_BAND_M[1] + P.ABOVE_GAP_M) & (z < P.ABOVE_MAX_M)])
            if len(c):
                above[c[:, 0], c[:, 1]] = True
                lo, hi = band.min(0), band.max(0) + 1
                above[lo[0]:hi[0], lo[1]:hi[1]] = ndimage.binary_dilation(above[lo[0]:hi[0], lo[1]:hi[1]], iterations=1)
            band = band[~above[band[:, 0], band[:, 1]]]
            self.top[band[:, 0], band[:, 1]] += 1.0
        if self.freeze_cabinet:
            return
        est = self._measure_cabinet(pts_odom)
        if est is None:
            return
        old = self.cabinet
        if old is not None and old.complete:
            d_new, d_old = abs(est.length - P.CAB_LENGTH_M), abs(old.length - P.CAB_LENGTH_M)
            if not est.complete or (not P.CAB_TRACK and d_new > P.CAB_LENGTH_GOOD_M and d_new > d_old - 0.01):
                old.seen_step = self.step              # a partial or merged view does not move a complete estimate
                return
            a = 1.0 if d_new < d_old - 0.01 else P.CAB_SMOOTH    # a view with an end seen better than before replaces the estimate
            est.centre = (1 - a) * old.centre + a * est.centre
            est.length = (1 - a) * old.length + a * est.length if d_new <= P.CAB_LENGTH_GOOD_M or a == 1.0 else old.length
            if old.face_measured and est.face_measured:
                est.face = (1 - a) * old.face + a * est.face
        self.cabinet = est
        self.counts['cabinet_frames'] += 1

    def _rect(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Rectangle of the cells: (centre, unit long axis, length, depth). Along the room axes once they are known (the cabinet stands at a
        wall), the extents from the 1-99 percentiles; else the minimum-area rectangle."""
        if self.room_psi is None:
            centre, ang, ext = G.min_area_rect(xy[:: max(1, len(xy) // 3000)])
            u = np.array([math.cos(ang), math.sin(ang)])
        else:
            u = np.array([math.cos(self.room_psi), math.sin(self.room_psi)])
            v = np.array([-u[1], u[0]])
            a, b = xy @ u, xy @ v
            (a0, a1), (b0, b1) = np.percentile(a, [1, 99]), np.percentile(b, [1, 99])
            centre = u * (a0 + a1) / 2 + v * (b0 + b1) / 2
            ext = np.array([a1 - a0, b1 - b0])
        ext = ext + P.TOP_RES_M
        if ext[1] > ext[0]:
            u, ext = np.array([-u[1], u[0]]), ext[::-1]
        return centre, u, float(ext[0]), float(ext[1])

    def _measure_cabinet(self, pts_odom: np.ndarray) -> Optional[CabinetEst]:
        mask = self.top >= P.TOP_MIN
        self.cab_diag = []
        if mask.sum() < 40:
            return None
        lab, _ = ndimage.label(ndimage.binary_closing(mask, iterations=1))
        base = self.pose()[:2]
        best, best_score = None, 1e9
        for k, sl in enumerate(ndimage.find_objects(lab), start=1):
            cells = np.argwhere(lab[sl] == k) + np.array([sl[0].start, sl[1].start])
            if len(cells) < 40:
                continue
            xy = (cells + 0.5) * P.TOP_RES_M - P.MAP_HALF_M
            centre, u, length, depth = self._rect(xy)
            fill = len(cells) * P.TOP_RES_M ** 2 / (length * depth)
            self.cab_diag.append([round(float(v), 3) for v in (*centre, length, depth, fill)])
            if abs(depth - P.CAB_DEPTH_M) > P.CAB_DEPTH_TOL or length < P.CAB_MIN_LENGTH_M or length > P.CAB_LENGTH_M + P.CAB_LENGTH_TOL:
                continue
            if fill < P.CAB_MIN_FILL:                   # chair backs at the same height form thin arcs, the top is a filled plate
                continue
            if self.cabinet is not None and self.cabinet.complete and np.linalg.norm(centre - self.cabinet.centre) > P.CAB_LOCK_M:
                continue
            score = abs(depth - P.CAB_DEPTH_M) + 0.3 * max(0., P.CAB_LENGTH_M - length) + 0.05 * float(np.linalg.norm(centre - base))
            if score < best_score:
                best, best_score = (centre, u, length, depth), score
        if best is None:
            return None
        centre, u, length, depth = best
        f = np.array([-u[1], u[0]])
        if float(np.dot(base - centre, f)) < 0:
            f = -f
        z = pts_odom[:, 2]
        top_pts = pts_odom[(z > P.CAB_TOP_BAND_M[0]) & (z < P.CAB_TOP_BAND_M[1])]
        rel = top_pts[:, :2] - centre
        near = top_pts[(np.abs(rel @ f) < depth / 2 - 0.03) & (np.abs(rel @ u) < length / 2 - 0.03)]
        top_z = float(np.median(near[:, 2])) if len(near) > 20 else (self.cabinet.top_z if self.cabinet is not None else float(np.mean(P.CAB_TOP_BAND_M)))
        est = CabinetEst(centre=centre, f=f, length=length, depth=depth, top_z=top_z, face=depth / 2 + P.DRAWER_FACE_OUT_M,
                         complete=length >= P.CAB_LENGTH_M - P.CAB_LENGTH_AGREE_M, seen_step=self.step)
        self._measure_face(est, pts_odom)
        return est

    def _measure_face(self, est: CabinetEst, pts_odom: np.ndarray) -> None:
        """The drawer face: points of the drawer band just in front of the top edge, inside the front span."""
        z = pts_odom[:, 2]
        p = pts_odom[(z > P.FACE_BAND_M[0]) & (z < P.FACE_BAND_M[1]), :2] - est.centre
        if not len(p):
            return
        s = p @ est.f - est.depth / 2
        lat = p @ est.lateral
        sel = (s > P.FACE_WINDOW_M[0]) & (s < P.FACE_WINDOW_M[1]) & (np.abs(lat) < est.length / 2 - 0.05)
        if sel.sum() >= P.FACE_MIN_POINTS:
            est.face = est.depth / 2 + float(np.median(s[sel]))
            est.face_measured = True
