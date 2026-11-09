"""The scanner of the fair planner of task 98: a rectangle of known size fitted to the top of the tall cluster, tracked over time, its button end
decided by the red blob.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from core.skills_lib import rgbd_geometry as G

from .. import params as P
from ..world import ScannerEst

def fit_rect(xy: np.ndarray, yaw: float, size: Tuple[float, float], prior_centre: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """Centre of a rectangle of known size (long, short) and orientation `yaw` (long axis direction) covering the points; when the points span
    less than the rectangle along an axis, the rectangle is placed closest to `prior_centre`. Returns (centre, coverage in [0, 1])."""
    ex, ey = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
    uv = np.stack([xy @ ex, xy @ ey], 1)
    lo, hi = np.percentile(uv, 1.5, axis=0), np.percentile(uv, 98.5, axis=0)
    pu = None if prior_centre is None else np.array([prior_centre @ ex, prior_centre @ ey])
    c, cover = np.zeros(2), 1.
    for k in (0, 1):
        span = hi[k] - lo[k]
        mid = (hi[k] + lo[k]) / 2
        if span >= size[k] * 0.93 or pu is None:
            c[k] = mid
        else:
            c[k] = float(np.clip(pu[k], hi[k] - size[k] / 2, lo[k] + size[k] / 2))
        cover = min(cover, span / size[k])
    return c[0] * ex + c[1] * ey, float(min(cover, 1.))


class ScannerMixin:
    def _scanner_key(self, cp: np.ndarray) -> float:
        """Preference among the tall clusters on the desk: the nearest to the tracked scanner, else the biggest (smaller key is better)."""
        prev = self.scanner
        if prev is not None and self.step - prev.seen_step < 300:
            return float(np.linalg.norm(cp[:, :2].mean(0) - prev.centre))
        return -float(len(cp))

    def _fit_scanner(self, cp: np.ndarray, h: float) -> None:
        zmax = float(np.percentile(cp[:, 2], 97))
        top = cp[cp[:, 2] > zmax - 0.02]
        size = (P.SCANNER_LONG_M, P.SCANNER_SHORT_M)
        prev = self.scanner
        centre, ang, ext = G.min_area_rect(top[:, :2])
        long_ok = abs(max(ext) - size[0]) < 0.05 and abs(min(ext) - size[1]) < 0.05
        if long_ok and prev is not None and np.linalg.norm(centre - prev.centre) > P.SCANNER_JUMP_M and self.step - prev.seen_step < 300:
            long_ok = False          # the scanner does not teleport: a different object, keep the track
        if long_ok:
            axis = ang if ext[0] >= ext[1] else ang + math.pi / 2
            yaw = axis if prev is None else (axis if abs(G.wrap(axis - prev.yaw)) <= math.pi / 2 else axis + math.pi)
            if prev is None and self.marker_blob is not None:   # the button decides the end
                u = (self.marker_blob[:2] - centre) @ np.array([math.cos(axis), math.sin(axis)])
                yaw = axis if u >= 0 else axis + math.pi
            front_known = prev.front_known if prev is not None else self.marker_blob is not None
            yaw = G.wrap(yaw)
            if prev is not None and self.step - prev.seen_step < P.SCANNER_TRACK_STEPS:     # a tracked scanner moves slowly: rate-limit the update
                yaw = G.wrap(prev.yaw + float(np.clip(G.wrap(yaw - prev.yaw), -P.SCANNER_YAW_STEP, P.SCANNER_YAW_STEP)))
                shift = centre - prev.centre
                n = float(np.linalg.norm(shift))
                centre = prev.centre + shift * min(1., P.SCANNER_SHIFT_STEP_M / max(n, 1e-9))
            self.scanner = ScannerEst(centre=centre, yaw=yaw, half=(size[0] / 2, size[1] / 2), z_lo=self.top_z, z_hi=zmax,
                                      seen_step=self.step, front_known=front_known)
            self.counts['scanner_frames'] += 1
        elif prev is not None and self.allow_partial and len(top) > P.SCANNER_MIN_POINTS and \
                np.linalg.norm(top[:, :2].mean(0) - prev.centre) < P.SCANNER_JUMP_M:
            axis = prev.yaw
            centre, cover = fit_rect(top[:, :2], axis, size, prev.centre)
            if cover > P.SCANNER_MIN_COVER:
                self.scanner = ScannerEst(centre=centre, yaw=prev.yaw, half=prev.half, z_lo=self.top_z, z_hi=zmax, seen_step=self.step,
                                          front_known=prev.front_known)
                self.counts['scanner_frames'] += 1
        self._decide_front()

    def _decide_front(self) -> None:
        """The marker blob lies on the button end: turn the scanner yaw to point at it."""
        s = self.scanner
        if s is None or self.marker_blob is None:
            return
        limit = P.BLOB_FRESH_STEPS if s.front_known else 3 * P.PERCEIVE_EVERY * 40
        if self.step - self.marker_blob_step > limit:
            return
        ex = np.array([math.cos(s.yaw), math.sin(s.yaw)])
        u = float((self.marker_blob[:2] - s.centre) @ ex)
        if abs(u) > 0.06 and np.linalg.norm(self.marker_blob[:2] - s.centre) < 0.3:
            if u < 0:
                s.yaw = G.wrap(s.yaw + math.pi)
            s.front_known = True
