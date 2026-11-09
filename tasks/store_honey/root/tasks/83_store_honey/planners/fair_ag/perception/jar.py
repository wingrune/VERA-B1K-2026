"""The jar for the fair planner of task 83: a small upright object (asset prior: JAR_DIAMETER_M wide, JAR_HEIGHT_M tall) standing on a raised
horizontal surface. Per perception, the points of all cameras are binned into JAR_RES_M cells of the odometry frame (highest point per cell);
cells about JAR_HEIGHT_M above the low quartile of the seen cells next to them are grouped into blobs; a blob of the jar's footprint and point spread whose
ring of cells around it is one flat surface raised above the floor is a jar sighting. Sightings are fused: the
group of sightings within JAR_GROUP_M with the most members is the jar, its estimate the median of the latest JAR_KEEP members.

Also kept: the height layer (top of the last points seen per cell), which the arm clearance check reads."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy import ndimage

from .. import params as P
from ..world import JarEst

JAR_LOCK_GATE_M = 0.3     # a group this far from the locked jar needs at least as many sightings to replace it


class JarMixin:
    def _height_cells(self, pts_odom: np.ndarray, res: float):
        c = np.floor((pts_odom[:, :2] + P.MAP_HALF_M) / res).astype(int)
        n = int(2 * P.MAP_HALF_M / res)
        ok = (c >= 0).all(1) & (c < n).all(1)
        return c[ok], pts_odom[ok, 2], n

    def _update_heights(self, pts_odom: np.ndarray) -> None:
        """Top of the points of this frame per cell overwrites the layer (the drawer and the jar move)."""
        z0 = pts_odom[:, 2]
        sel = (z0 > P.FLOOR_BAND_M) & (z0 < P.ABOVE_MAX_M)
        c, z, n = self._height_cells(pts_odom[sel], P.HEIGHT_RES_M)
        if not len(z):
            return
        flat = c[:, 0] * n + c[:, 1]
        top = np.full(n * n, -np.inf, np.float32)
        np.maximum.at(top, flat, z.astype(np.float32))
        seen = np.isfinite(top)
        self.heights.reshape(-1)[seen] = top[seen]

    def top_height(self, xy: np.ndarray) -> Optional[float]:
        """Top of the last points seen in the height cell of xy (odometry frame), None when the cell was never seen."""
        n = self.heights.shape[0]
        i, j = (np.floor((np.asarray(xy, float)[:2] + P.MAP_HALF_M) / P.HEIGHT_RES_M)).astype(int)
        if not (0 <= i < n and 0 <= j < n):
            return None
        v = float(self.heights[i, j])
        return v if np.isfinite(v) else None

    def _sight_jars(self, pts_odom: np.ndarray, eye_xy: np.ndarray) -> List[dict]:
        """Jar sightings in one frame: dicts (xy, top, support)."""
        z0 = pts_odom[:, 2]
        pts = pts_odom[(z0 > P.JAR_SUPPORT_MIN_M) & (z0 < P.JAR_TOP_MAX_M)]
        d = np.linalg.norm(pts[:, :2] - eye_xy, axis=1)
        pts = pts[d < min(P.JAR_RANGE_M, 3.2)]
        if len(pts) < 20:
            return []
        res = P.JAR_RES_M
        lo = pts[:, :2].min(0) - 3 * res
        cells = np.floor((pts[:, :2] - lo) / res).astype(int)
        shape = tuple(cells.max(0) + 4)
        top = np.full(shape, np.nan)
        flat = np.full(shape[0] * shape[1], -np.inf)
        np.maximum.at(flat, cells[:, 0] * shape[1] + cells[:, 1], pts[:, 2])
        top = flat.reshape(shape)
        known = np.isfinite(top)
        top[~known] = np.nan
        # cells standing about a jar height above the lowest seen cell next to them (the surface the jar stands on)
        filled = np.where(known, top, np.inf)
        low = ndimage.percentile_filter(filled, P.JAR_LOW_PERCENTILE, size=2 * P.JAR_RING_CELLS + 1)
        step = top - low
        cand = known & np.isfinite(low) & (step > P.JAR_HEIGHT_M - P.JAR_HEIGHT_TOL_M) & (step < P.JAR_HEIGHT_M + P.JAR_HEIGHT_TOL_M)
        lab, _ = ndimage.label(cand, structure=np.ones((3, 3)))
        out = []
        max_cells = int((P.JAR_DIAMETER_M + 2 * res) ** 2 / res ** 2 * 1.6) + 2
        for k, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None:
                continue
            blob = lab[sl] == k
            ncell = int(blob.sum())
            ext = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) * res
            if ncell < 2 or ncell > max_cells or ext > P.JAR_DIAMETER_M + 3 * res:
                continue
            full = np.zeros(shape, bool)
            full[sl] = blob
            ring = ndimage.binary_dilation(full, iterations=P.JAR_RING_CELLS) & ~ndimage.binary_dilation(full, iterations=1)
            ring_known = ring & known
            if ring_known.sum() < max(4, P.JAR_RING_KNOWN * ring.sum()):
                continue
            rz = top[ring_known]
            support = float(np.percentile(rz, 50))
            jar_top = float(np.max(top[full]))
            h = jar_top - support
            if not (P.JAR_HEIGHT_M - P.JAR_HEIGHT_TOL_M <= h <= P.JAR_HEIGHT_M + P.JAR_HEIGHT_TOL_M) or support < P.JAR_SUPPORT_MIN_M:
                continue
            if float(np.mean(np.abs(rz - support) < P.PATCH_STEP_M)) < P.JAR_RING_FLAT:     # the ring is one surface (a counter, a table)
                continue
            ij = np.argwhere(full)
            sel = np.isin(cells[:, 0] * shape[1] + cells[:, 1], ij[:, 0] * shape[1] + ij[:, 1]) & (pts[:, 2] > support + 0.01)
            if sel.sum() < 5 or ncell < P.JAR_MIN_CELLS:
                continue
            spread = float(np.sqrt(np.mean(np.sum((pts[sel, :2] - pts[sel, :2].mean(0)) ** 2, 1))))
            if not (P.JAR_SPREAD_M[0] <= spread <= P.JAR_SPREAD_M[1]):     # the points of an 8 cm cylinder lie about 2.5 cm from its axis
                continue
            xy = pts[sel, :2].mean(0)
            away = xy - eye_xy
            xy = xy + away / max(float(np.linalg.norm(away)), 1e-6) * P.JAR_VIEW_SHIFT_M     # the seen side is nearer than the axis
            out.append(dict(xy=xy, top=jar_top, support=support, n=int(sel.sum()), cells=ncell, ext=round(ext, 3), spread=round(spread, 4)))
        return out

    def _update_jar(self, pts_odom: np.ndarray, eye_xy: np.ndarray) -> None:
        if self.freeze_jar:
            return
        for s in self._sight_jars(pts_odom, eye_xy):
            s['step'] = self.step
            self.jar_sightings.append(s)
        if len(self.jar_sightings) > P.JAR_MAX_SIGHTINGS:
            self.jar_sightings = self.jar_sightings[-P.JAR_MAX_SIGHTINGS:]
        self.jar = self._fuse_jar()

    def _fuse_jar(self) -> Optional[JarEst]:
        s = self.jar_sightings
        if not s:
            return self.jar_lock
        xy = np.array([v['xy'] for v in s])
        d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
        weight = np.sqrt(np.array([v['n'] for v in s], float))             # close views have many points
        score = ((d < P.JAR_GROUP_M) * weight[None]).sum(1)
        best = int(np.argmax(score + 1e-6 * np.arange(len(s))))           # ties: the latest
        group = [v for v, near in zip(s, d[best] < P.JAR_GROUP_M) if near][-P.JAR_KEEP:]
        if len(group) < min(P.JAR_MIN_SIGHTINGS, getattr(self, 'jar_min', P.JAR_MIN_SIGHTINGS)):
            return self.jar_lock
        c = np.median(np.array([v['xy'] for v in group]), 0)
        support = float(np.median([v['support'] for v in group]))
        top = float(np.median([v['top'] for v in group]))
        est = JarEst(xy=c, support=support, top=top, sightings=len(group), seen_step=max(v['step'] for v in group))
        if self.jar_lock is not None and np.linalg.norm(est.xy - self.jar_lock.xy) > JAR_LOCK_GATE_M \
                and est.sightings < self.jar_lock_n:
            return self.jar_lock                      # a sparse far group (a decoy) does not replace the locked jar
        self.jar_lock = est
        self.jar_lock_n = max(self.jar_lock_n, est.sightings)
        return est

    def forget_jar(self, keep_after: int) -> None:
        """Drop the sightings older than step `keep_after` (a fresh close look replaces the estimate made from far)."""
        self.jar_sightings = [v for v in self.jar_sightings if v['step'] >= keep_after]
        self.jar = self._fuse_jar()
