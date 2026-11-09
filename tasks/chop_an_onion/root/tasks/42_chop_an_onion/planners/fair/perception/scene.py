"""Counter and sink basin from the recent cloud (odometry frame, heights above the floor): top-down rasters and plane peaks, no object
instance numbers, no simulator state. Counter: the largest slab at the most populated height between COUNTER_BAND_M. Basin: the cells inside the
counter rectangle that are seen sunk below the top, their oriented rectangle."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

from core.skills_lib import rgbd_geometry as G

from .. import params as P
from ..world import BasinEst, CounterEst


class Raster:
    """Max-height raster of points.
    Max-height raster of points (N, 3) (heights already above the floor) on a square grid of cell `res` over their bounding box (or over the
    given `like` raster's grid)."""

    def __init__(self, pts: np.ndarray, res: float, pad: float = 0.0, like: Optional['Raster'] = None) -> None:
        self.res = res
        if like is not None:
            self.lo, self.shape = like.lo, like.shape
        else:
            self.lo = pts[:, :2].min(0) - pad
            n = np.floor((pts[:, :2].max(0) + pad - self.lo) / res).astype(int) + 1
            self.shape = (int(n[0]), int(n[1]))
        self.zmax = np.full(self.shape, -np.inf)
        self.count = np.zeros(self.shape, int)
        if len(pts):
            idx = self.index(pts[:, :2])
            np.maximum.at(self.zmax, (idx[:, 0], idx[:, 1]), pts[:, 2])
            np.add.at(self.count, (idx[:, 0], idx[:, 1]), 1)

    def index(self, xy: np.ndarray) -> np.ndarray:
        return np.clip(np.floor((np.asarray(xy, float) - self.lo) / self.res).astype(int), 0, np.array(self.shape) - 1)

    def centres(self, idx: np.ndarray) -> np.ndarray:
        return self.lo + (np.asarray(idx, float) + 0.5) * self.res

    @property
    def seen(self) -> np.ndarray:
        return self.count > 0


def fine_rect(xy: np.ndarray):
    """Refined min-area rectangle.
    G.min_area_rect (2 degree steps) refined to 0.25 degree steps around its angle: (centre, yaw, extents)."""
    centre, yaw, ext = G.min_area_rect(xy)
    best = None
    for da in np.radians(np.arange(-2.0, 2.01, 0.25)):
        a = yaw + da
        c, s_ = math.cos(a), math.sin(a)
        u = xy @ np.array([c, s_])
        v = xy @ np.array([-s_, c])
        e = np.array([u.max() - u.min(), v.max() - v.min()])
        if best is None or e[0] * e[1] < best[0]:
            best = (e[0] * e[1], a, 0.5 * (u.max() + u.min()), 0.5 * (v.max() + v.min()), e)
    _, a, cu, cv, e = best
    c, s_ = math.cos(a), math.sin(a)
    return cu * np.array([c, s_]) + cv * np.array([-s_, c]), a, e


def largest(mask: np.ndarray, min_cells: int) -> Optional[np.ndarray]:
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return None
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    return (lab == k) if sizes[k - 1] >= min_cells else None


def find_counter(pts, viewpoint, step, exclude=None) -> Optional[Tuple[CounterEst, Raster, np.ndarray]]:
    """Counter slab from points.
    Counter slab from points (N, 3) (xy odometry, z above the floor). `viewpoint` (xy) orients the normal towards the viewer.
    Returns (CounterEst, raster, mask of the slab) or None."""
    band = pts[(pts[:, 2] > P.COUNTER_BAND_M[0]) & (pts[:, 2] < P.COUNTER_BAND_M[1])]
    if len(band) < P.COUNTER_MIN_POINTS:
        return None
    top = G.height_peak(band[:, 2], P.COUNTER_BAND_M[0], P.COUNTER_BAND_M[1], 0.005, P.COUNTER_MIN_POINTS)
    if top is None:
        return None
    sel = band[np.abs(band[:, 2] - top) < P.COUNTER_FLAT_M]
    if len(sel) < P.COUNTER_MIN_POINTS:
        return None
    r = Raster(sel, P.COUNTER_RES_M, pad=0.05)
    occ = r.zmax > -np.inf
    occ = ndimage.binary_closing(occ, structure=np.ones((3, 3)), iterations=2)
    for ex in exclude or []:                      # counters already looked over without a sink in them
        cells = r.centres(np.argwhere(occ))
        d = cells - ex['centre']
        inside = (np.abs(d @ ex['u']) < 0.5 * ex['length'] + 0.2) & (np.abs(d @ ex['n']) < 0.5 * ex['depth'] + 0.2)
        idx = np.argwhere(occ)[inside]
        occ[idx[:, 0], idx[:, 1]] = False
    comp = largest(occ, P.COUNTER_MIN_AREA_M2 / P.COUNTER_RES_M ** 2)
    if comp is None:
        return None
    for _ in range(2):                  # other flat things at the same height (hands, a shelf) hang on by thin links: open them off
        idx = np.argwhere(comp)
        xy = r.centres(idx[::max(1, len(idx) // 3000)])
        centre, yaw, ext = G.min_area_rect(xy)
        if max(ext) < 0.5 or min(ext) <= P.COUNTER_MAX_DEPTH_M:
            break
        opened = ndimage.binary_opening(comp, structure=np.ones((P.COUNTER_OPEN_CELLS, P.COUNTER_OPEN_CELLS)))
        comp = largest(opened, P.COUNTER_MIN_AREA_M2 / P.COUNTER_RES_M ** 2)
        if comp is None:
            return None
    idx = np.argwhere(comp)
    xy = r.centres(idx[::max(1, len(idx) // 3000)])
    centre, yaw, ext = fine_rect(xy)
    u = np.array([math.cos(yaw), math.sin(yaw)])
    v = np.array([-math.sin(yaw), math.cos(yaw)])
    if ext[0] < ext[1]:
        u, v, ext = v, -u, ext[::-1]
    n = v if float((np.asarray(viewpoint, float)[:2] - centre) @ v) >= 0 else -v
    if ext[1] < P.COUNTER_MIN_DEPTH_M or ext[1] > P.COUNTER_MAX_DEPTH_M or ext[0] < P.COUNTER_MIN_LENGTH_M:
        return None
    # the rectangle of the slab; the front edge is the side the viewer sees
    est = CounterEst(top=float(top), centre=np.asarray(centre, float), u=u, n=n, length=float(ext[0]), depth=float(ext[1]), seen_step=step)
    return est, r, comp


def local_points(pts: np.ndarray, counter: CounterEst) -> np.ndarray:
    """Points in the counter frame.
    (a, b, h): coordinates along the counter's long side and its normal, and the height above the floor."""
    d = pts[:, :2] - counter.centre
    return np.c_[d @ counter.u, d @ counter.n, pts[:, 2]]


def find_basin(pts: np.ndarray, counter: CounterEst, step: int) -> Optional[BasinEst]:
    """Basin by template matching.
    The basin by template matching in the counter frame (axis-aligned with the counter: the sink is built into it, its long side across it).
    Evidence per 1 cm cell: floor (a point at basin depth) and top (a point at the counter height; overhead things such as the faucet give no top
    evidence). A box of the prior size scores its floor cells, loses for top cells inside it and for floor cells just outside, and gains for top
    cells just outside (the visible rim it touches). Cells the camera cannot see (the front part of the floor hides behind the rim) count for
    nothing, so the box is pushed against the rim that is visible."""
    from scipy.signal import fftconvolve
    loc = local_points(pts, counter)
    inside = (np.abs(loc[:, 0]) < 0.5 * counter.length + 0.05) & (np.abs(loc[:, 1]) < 0.5 * counter.depth + 0.05)
    cand = loc[inside]
    if len(cand) < 100:
        return None
    low = cand[(cand[:, 2] < counter.top - P.BASIN_BELOW_M[0]) & (cand[:, 2] > counter.top - P.BASIN_BELOW_M[1])]
    if len(low) < P.BASIN_MIN_POINTS:
        return None
    top_pts = cand[np.abs(cand[:, 2] - counter.top) < P.COUNTER_FLAT_M]
    r_low = Raster(low, P.BASIN_RES_M, pad=0.25)
    r_top = Raster(top_pts, P.BASIN_RES_M, like=r_low)
    floor = ndimage.binary_opening(r_low.seen, structure=np.ones((2, 2))).astype(float)
    top = r_top.seen.astype(float)
    wa, wb = P.BASIN_SIZE_M
    ka, kb = int(round(wa / P.BASIN_RES_M)), int(round(wb / P.BASIN_RES_M))
    kin = np.ones((ka, kb))
    pad = 3
    kout = np.ones((ka + 2 * pad, kb + 2 * pad))
    kout[pad:-pad, pad:-pad] = 0.0
    score = (fftconvolve(floor, kin[::-1, ::-1], mode='same') - P.BASIN_TOP_IN * fftconvolve(top, kin[::-1, ::-1], mode='same')
             - P.BASIN_FLOOR_OUT * fftconvolve(floor, kout[::-1, ::-1], mode='same') + P.BASIN_TOP_RIM * fftconvolve(top, kout[::-1, ::-1], mode='same'))
    k = np.unravel_index(int(np.argmax(score)), score.shape)
    if float(score[k]) < P.BASIN_MIN_SCORE:
        return None
    patch = floor[max(0, k[0] - ka // 2):k[0] + ka - ka // 2, max(0, k[1] - kb // 2):k[1] + kb - kb // 2]
    lab_p, n_p = ndimage.label(patch > 0, structure=np.ones((3, 3)))
    if n_p > 1:                     # a real basin floor seen from the front is one compact patch; a bogus match straddles fragments
        sizes = np.bincount(lab_p.ravel())[1:]
        if float(sizes.max()) < 0.6 * float(patch.sum()):
            return None
    ca, cb = r_low.centres(np.array(k))
    zc = r_low.zmax[floor > 0]
    fl = G.height_peak(zc, counter.top - P.BASIN_BELOW_M[1], counter.top - P.BASIN_BELOW_M[0], 0.005, 20)
    fl = float(np.percentile(zc, 20)) if fl is None else float(fl)
    return BasinEst(centre=counter.from_local(ca, cb), yaw=math.atan2(counter.u[1], counter.u[0]), size=np.array([wa, wb]), floor=fl, seen_step=step)
