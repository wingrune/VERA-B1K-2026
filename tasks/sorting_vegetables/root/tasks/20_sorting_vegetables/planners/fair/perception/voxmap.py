"""Occupancy of what the head camera saw, in the odometry frame: a voxel grid of hit counts (for the arm's body checks) and its projection
on the floor in the base's height band (for routes). Hits fade a little at every update, so what is no longer seen (a vegetable moved, a
pose correction) disappears after a while; what the head keeps seeing stays. Floor cells seen empty are kept as `floor_seen`."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

from .. import params as P


class VoxMap:
    def __init__(self) -> None:
        n = int(round(2 * P.MAP_HALF_M / P.VOX_M))
        nz = int(round(P.VOX_TOP_M / P.VOX_M))
        self.hits = np.zeros((n, n, nz), np.float32)
        self.floor_seen = np.zeros((n, n), bool)
        self.lo = np.array([-P.MAP_HALF_M, -P.MAP_HALF_M, 0.])
        self.updates = 0
        self._dil = None

    # ---- cells --------------------------------------------------------------------------------------------------------------------------
    def cells(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(voxel indices (N, 3), mask of points inside the grid)."""
        idx = np.floor((np.asarray(pts, float) - self.lo) / P.VOX_M).astype(int)
        ok = np.all(idx >= 0, axis=1) & np.all(idx < np.array(self.hits.shape), axis=1)
        return idx, ok

    def cell2(self, xy) -> Tuple[int, int]:
        i = np.clip(np.floor((np.asarray(xy, float)[:2] - self.lo[:2]) / P.VOX_M).astype(int), 0, self.hits.shape[0] - 1)
        return int(i[0]), int(i[1])

    def centre2(self, idx: np.ndarray) -> np.ndarray:
        return (np.asarray(idx, float) + 0.5) * P.VOX_M + self.lo[:2]

    # ---- updates ------------------------------------------------------------------------------------------------------------------------
    def carve(self, pts: np.ndarray, origin: np.ndarray, max_rays: int = 400, margin: float = 0.12) -> None:
        """Free-space carving: every voxel a head ray crossed before its hit (all but the last `margin` m) loses one hit, so a phantom
        obstacle (pose correction, moved vegetable) that the camera sees through is forgotten at once, while a wall hit again each view stays."""
        o = np.asarray(origin, float)
        if len(pts) == 0:
            return
        if len(pts) > max_rays:
            pts = pts[::len(pts) // max_rays + 1]
        d = pts - o
        L = np.linalg.norm(d, axis=1)
        ok = (L > 0.6) & np.isfinite(L)
        if not ok.any():
            return
        d, L = d[ok], L[ok]
        u = d / L[:, None]
        step = 0.5 * P.VOX_M
        s = np.arange(0.3, float(L.max()), step)
        if len(s) == 0:
            return
        valid = s[None, :] < (L[:, None] - margin)
        p3 = o + u[:, None, :] * s[None, :, None]
        p3 = p3[valid]
        if len(p3) == 0:
            return
        idx, inside = self.cells(p3)
        idx = np.unique(idx[inside], axis=0)
        if len(idx):
            h = self.hits[idx[:, 0], idx[:, 1], idx[:, 2]]
            self.hits[idx[:, 0], idx[:, 1], idx[:, 2]] = np.maximum(h - 1.0, 0.0)

    def add(self, pts: np.ndarray, floor_z: float = 0.0, origin: Optional[np.ndarray] = None) -> None:
        """Head points (odometry frame, z above the floor): hits above the floor band, floor sightings at it."""
        self.hits *= P.VOX_DECAY
        self.updates += 1
        if origin is not None:
            self.carve(pts, origin)
        z = pts[:, 2] - floor_z
        above = pts[z > P.FLOOR_BAND_M]
        idx, ok = self.cells(above)
        idx = idx[ok]
        if len(idx):
            idx = np.unique(idx, axis=0)
            self.hits[idx[:, 0], idx[:, 1], idx[:, 2]] = np.minimum(self.hits[idx[:, 0], idx[:, 1], idx[:, 2]] + 1.0, P.VOX_MAX_HITS)
        fl = pts[np.abs(z) <= P.FLOOR_BAND_M]
        if len(fl):
            i2, ok2 = self.cells(np.c_[fl[:, :2], np.zeros(len(fl))])
            i2 = i2[ok2]
            self.floor_seen[i2[:, 0], i2[:, 1]] = True

    def clear_box(self, lo: np.ndarray, hi: np.ndarray) -> None:
        """Forget the hits in an odometry box (an object that was moved by the robot)."""
        a, _ = self.cells(np.asarray(lo, float)[None])
        b, _ = self.cells(np.asarray(hi, float)[None])
        a = np.clip(a[0], 0, np.array(self.hits.shape) - 1)
        b = np.clip(b[0], 0, np.array(self.hits.shape) - 1)
        self.hits[a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1] = 0.
        self.updates += 1

    # ---- products -----------------------------------------------------------------------------------------------------------------------
    def occupied(self) -> np.ndarray:
        return self.hits >= P.VOX_MIN_HITS

    def blocked2d(self, band: Tuple[float, float] = None) -> np.ndarray:
        """Floor cells with an occupied voxel in the base's height band."""
        lo, hi = band or P.NAV_BAND_M
        k0, k1 = int(lo / P.VOX_M), int(np.ceil(hi / P.VOX_M))
        return self.occupied()[:, :, k0:k1].any(axis=2)

    def points_hit(self, pts: np.ndarray, margin_cells: int = 0, ignore: Optional[list] = None) -> np.ndarray:
        """Per point: an occupied voxel within margin_cells of it (the dilated occupancy is cached until the next update). Points inside an
        `ignore` box (lo, hi, odometry) do not count (the object the hand goes to)."""
        key = (self.updates, margin_cells)
        if self._dil is None or self._dil[0] != key:
            occ = self.occupied()
            self._dil = (key, ndimage.binary_dilation(occ, iterations=margin_cells) if margin_cells > 0 else occ)
        occ = self._dil[1]
        idx, ok = self.cells(pts)
        out = np.zeros(len(pts), bool)
        out[ok] = occ[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
        for lo, hi in ignore or ():
            out &= ~np.all((pts >= np.asarray(lo, float)) & (pts <= np.asarray(hi, float)), axis=1)
        return out

    def height_at(self, xy: np.ndarray, below: float) -> float:
        """Highest occupied voxel top under `below` at the floor cell of xy (0 for none)."""
        i, j = self.cell2(xy)
        col = self.occupied()[i, j, :int(below / P.VOX_M)]
        k = np.flatnonzero(col)
        return float((k[-1] + 1) * P.VOX_M) if len(k) else 0.0
