"""The task items on and in the counter from the recent cloud: components of the top-down raster above the counter top classified by their
footprint and height (board: flat and wide, parer: flat and thin, bowl: round and 5-11 cm high), the onion (a blob inside the basin), the pieces
of the cut onion (blobs inside the bowl). Shape priors are the asset sizes (data/models.npz); there is no colour and no instance number."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import ndimage

from core.skills_lib import rgbd_geometry as G

from .. import params as P
from ..world import BasinEst, CounterEst, ItemEst
from .scene import Raster, local_points


def _components(r: Raster, mask: np.ndarray, min_cells: int) -> List[np.ndarray]:
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    out = []
    for k in range(1, n + 1):
        idx = np.argwhere(lab == k)
        if len(idx) >= min_cells:
            out.append(idx)
    return out


def _rect(r: Raster, idx: np.ndarray):
    """Cell rectangle.
    (centre (a, b), yaw, extents (long, short) with yaw along the long one) of the cells."""
    xy = r.centres(idx[::max(1, len(idx) // 1500)])
    c, yaw, ext = G.min_area_rect(xy)
    if ext[1] > ext[0]:
        yaw, ext = yaw + math.pi / 2, ext[::-1]
    return np.asarray(c, float), float(yaw), np.asarray(ext, float) + r.res


def classify(ext: np.ndarray, hmax: float) -> Optional[str]:
    long_, short = float(ext[0]), float(ext[1])
    if hmax <= P.FLAT_MAX_M:
        if short <= P.PARER_SHORT_MAX_M and P.PARER_LONG_M[0] <= long_ <= P.PARER_LONG_M[1]:
            return 'parer'
        if short >= P.BOARD_SHORT_MIN_M and P.BOARD_LONG_M[0] <= long_ <= P.BOARD_LONG_M[1]:
            return 'board'
        return None
    if P.BOWL_H_M[0] <= hmax <= P.BOWL_H_M[1] and short >= P.BOWL_SHORT_MIN_M and long_ <= P.BOWL_LONG_MAX_M:
        return 'bowl'
    return None


def _parer_fine(sel: np.ndarray, counter: CounterEst, step: int, avoid: list) -> List[ItemEst]:
    """Second pass for the parer when the standard pass found none.
    The parer is thin and flat: from the vantage distance its footprint gives fewer
    cells than ITEM_MIN_CELLS at ITEM_RES_M, so it is not classified. This pass
    rasterises at half the cell size and accepts small components; discs around the
    board and the bowl already found are skipped (their rims are also thin)."""
    out: List[ItemEst] = []
    r = Raster(sel[:, [0, 1, 2]], 0.5 * P.ITEM_RES_M, pad=0.01)
    mask = ndimage.binary_closing(r.seen, structure=np.ones((3, 3)))
    ia = (np.arange(r.shape[0]) * r.res + r.lo[0] + 0.5 * r.res)[:, None]
    ib = (np.arange(r.shape[1]) * r.res + r.lo[1] + 0.5 * r.res)[None, :]
    for c2, rad in avoid:
        mask &= np.hypot(ia - c2[0], ib - c2[1]) > rad
    for idx in _components(r, mask, 5):
        c, yaw, ext = _rect(r, idx)
        if not (P.PARER_LONG_M[0] <= float(ext[0]) <= P.PARER_LONG_M[1]) or float(ext[1]) > P.PARER_SHORT_MAX_M:
            continue
        zs = r.zmax[idx[:, 0], idx[:, 1]]
        hmax = float(np.percentile(zs[np.isfinite(zs)], 95)) - counter.top
        if hmax > P.FLAT_MAX_M + 0.02 or hmax < 0.004:
            continue
        xy = counter.from_local(*c)
        ang = math.atan2(counter.u[1], counter.u[0]) + yaw
        out.append(ItemEst(role='parer', centre=np.array([xy[0], xy[1], counter.top + 0.5 * hmax]),
                           top=counter.top + hmax, size=np.array([ext[0], ext[1], hmax]), yaw=ang,
                           n_points=int(len(idx)), seen_step=step))
    return out


def _board_fine(sel: np.ndarray, counter: CounterEst, basin: BasinEst, step: int, avoid: list) -> List[ItemEst]:
    """Second pass for the board when the standard pass found none.
    From the vantage distance a board partly hidden behind the bowl (or clipped by
    the basin margin) shows as a flat strip too narrow for BOARD_SHORT_MIN_M, so it
    is never classified and the sink stage sees no board at all (302: 'missing
    board' at every look, then a grasp at a stale false estimate). This pass
    rasterises at half the cell size, cuts the basin box, skips discs around the
    parer and the bowl already found, and accepts flat components whose long side
    is board-sized and whose short side is at least 6 cm - a strip narrower than a
    real board must still have a board-sized footprint area."""
    out: List[ItemEst] = []
    r = Raster(sel[:, [0, 1, 2]], 0.5 * P.ITEM_RES_M, pad=0.01)
    mask = ndimage.binary_closing(r.seen, structure=np.ones((3, 3)))
    ia = (np.arange(r.shape[0]) * r.res + r.lo[0] + 0.5 * r.res)[:, None]
    ib = (np.arange(r.shape[1]) * r.res + r.lo[1] + 0.5 * r.res)[None, :]
    for c2, rad in avoid:
        mask &= np.hypot(ia - c2[0], ib - c2[1]) > rad
    if basin is not None:
        ba, bb = counter.to_local(basin.centre)
        wa, wb = basin.size
        mask &= ~((np.abs(ia - ba) < 0.5 * wa + P.SINK_MARGIN_M) & (np.abs(ib - bb) < 0.5 * wb + P.SINK_MARGIN_M))
    for idx in _components(r, mask, 12):
        c, yaw, ext = _rect(r, idx)
        long_, short = float(ext[0]), float(ext[1])
        if not P.BOARD_LONG_M[0] <= long_ <= P.BOARD_LONG_M[1] + 0.04 or not 0.06 <= short <= 0.34:
            continue
        if short < P.BOARD_SHORT_MIN_M and len(idx) * (0.5 * P.ITEM_RES_M) ** 2 < 0.012:
            continue
        zs = r.zmax[idx[:, 0], idx[:, 1]]
        hmax = float(np.percentile(zs[np.isfinite(zs)], 95)) - counter.top
        if hmax > P.FLAT_MAX_M + 0.02 or hmax < 0.004:
            continue
        xy = counter.from_local(*c)
        ang = math.atan2(counter.u[1], counter.u[0]) + yaw
        out.append(ItemEst(role='board', centre=np.array([xy[0], xy[1], counter.top + 0.5 * hmax]),
                           top=counter.top + hmax, size=np.array([long_, short, hmax]), yaw=ang,
                           n_points=int(len(idx)), seen_step=step))
    return out


def find_items(pts, counter, basin, step, skip=None
               ) -> Dict[str, List[ItemEst]]:
    """Item candidates per role.
    Candidates per role (best first by point count). pts (N, 3): odometry xy, height above the floor. `skip` lists xy discs (centre, radius)
    that are not items (the sink region is skipped by the basin box)."""
    loc = local_points(pts, counter)
    h = loc[:, 2] - counter.top
    inside = (np.abs(loc[:, 0]) < 0.5 * counter.length + 0.03) & (np.abs(loc[:, 1]) < 0.5 * counter.depth + 0.03)
    sel = loc[inside & (h > P.ITEM_BAND_M[0]) & (h < P.ITEM_BAND_M[1])]
    out: Dict[str, List[ItemEst]] = {'board': [], 'parer': [], 'bowl': []}
    if len(sel) < 30:
        return out
    r = Raster(sel[:, [0, 1, 2]], P.ITEM_RES_M, pad=0.02)
    mask = ndimage.binary_closing(r.seen, structure=np.ones((3, 3)))
    if basin is not None:                                     # the sink region is not counter: cut the basin box (plus the rim)
        ba, bb = counter.to_local(basin.centre)
        wa, wb = basin.size
        ia = (np.arange(r.shape[0]) * r.res + r.lo[0] + 0.5 * r.res)[:, None]
        ib = (np.arange(r.shape[1]) * r.res + r.lo[1] + 0.5 * r.res)[None, :]
        mask &= ~((np.abs(ia - ba) < 0.5 * wa + P.SINK_MARGIN_M) & (np.abs(ib - bb) < 0.5 * wb + P.SINK_MARGIN_M))
    for idx in _components(r, mask, P.ITEM_MIN_CELLS):
        c, yaw, ext = _rect(r, idx)
        zs = r.zmax[idx[:, 0], idx[:, 1]]
        hmax = float(np.percentile(zs[np.isfinite(zs)], 95)) - counter.top
        role = classify(ext, hmax)
        if role is None:
            continue
        cell_pts = r.count[idx[:, 0], idx[:, 1]].sum()
        xy = counter.from_local(*c)
        # yaw of the long side in the odometry frame
        ang = math.atan2(counter.u[1], counter.u[0]) + yaw
        top = counter.top + hmax
        out[role].append(ItemEst(role=role, centre=np.array([xy[0], xy[1], counter.top + 0.5 * hmax]), top=top, size=np.array([ext[0], ext[1], hmax]),
                                 yaw=ang, n_points=int(cell_pts), seen_step=step))
    if not out['parer'] and len(sel) >= 30:                   # thin parer missed at ITEM_RES_M: one finer pass
        avoid = [(counter.to_local(e.centre[:2]), 0.5 * max(float(e.size[0]), float(e.size[1])) + 0.02)
                 for e in out['board'] + out['bowl']]
        out['parer'] = _parer_fine(sel, counter, step, avoid)
    if not out['board'] and len(sel) >= 30:                   # occluded board missed at ITEM_RES_M: one finer pass
        avoid = [(counter.to_local(e.centre[:2]), 0.5 * max(float(e.size[0]), float(e.size[1])) + 0.02)
                 for e in out['parer'] + out['bowl']]
        out['board'] = _board_fine(sel, counter, basin, step, avoid)
    for lst in out.values():
        lst.sort(key=lambda e: -e.n_points)
    return out


def find_onion(pts: np.ndarray, counter: CounterEst, basin: BasinEst, step: int) -> Optional[ItemEst]:
    """The onion blob in the basin."""
    loc = local_points(pts, counter)
    ba, bb = counter.to_local(basin.centre)
    inside = (np.abs(loc[:, 0] - ba) < 0.5 * basin.size[0] - P.ONION_WALL_M) & (np.abs(loc[:, 1] - bb) < 0.5 * basin.size[1] - P.ONION_WALL_M)
    sel = loc[inside & (loc[:, 2] > basin.floor + P.ONION_MIN_ABOVE_FLOOR_M) & (loc[:, 2] < counter.top - 0.02)]
    if len(sel) < P.ONION_MIN_POINTS:
        return None
    r = Raster(sel, P.ITEM_RES_M, pad=0.01)
    comps = _components(r, ndimage.binary_closing(r.seen, structure=np.ones((3, 3))), 8)
    if not comps:
        return None
    best = max(comps, key=len)
    zs = r.zmax[best[:, 0], best[:, 1]]
    top = float(np.max(zs[np.isfinite(zs)]))
    cap = best[zs > top - P.ONION_CAP_M]                       # the upper cap of the onion: its centre in xy, its top gives the centre height
    xy = counter.from_local(*r.centres(cap).mean(0))
    size = np.array([2 * P.ONION_R_M, 2 * P.ONION_R_M, 2 * P.ONION_R_M])
    return ItemEst(role='onion', centre=np.array([xy[0], xy[1], top - P.ONION_R_M]), top=top, size=size, n_points=int(len(sel)), seen_step=step)


def find_pieces(pts: np.ndarray, counter: CounterEst, bowl: ItemEst, step: int) -> Dict[str, Any]:
    """Points lying in the bowl.
    What lies in the bowl: points inside its footprint more than PIECE_MIN_ABOVE_M over the bowl's inner floor. Returns n points, the highest
    point (height above the counter top), centroid xy, and the xy of each blob (two halves = two blobs)."""
    loc = local_points(pts, counter)
    c = counter.to_local(bowl.centre[:2])
    inside = np.linalg.norm(loc[:, :2] - c, axis=1) < P.BOWL_INNER_R_M
    sel = loc[inside & (loc[:, 2] > counter.top + P.BOWL_FLOOR_M + P.PIECE_MIN_ABOVE_M) & (loc[:, 2] < counter.top + 0.2)]
    out: Dict[str, Any] = dict(n=int(len(sel)), top=0.0, xy=None, blobs=[], step=step)
    if len(sel) < P.PIECE_MIN_POINTS:
        return out
    r = Raster(sel, P.ITEM_RES_M, pad=0.01)
    comps = [c_ for c_ in _components(r, ndimage.binary_closing(r.seen, structure=np.ones((3, 3))), 6)]
    out['top'] = float(sel[:, 2].max() - counter.top)
    out['xy'] = counter.from_local(*sel[:, :2].mean(0)).tolist()
    for idx in comps:
        z = r.zmax[idx[:, 0], idx[:, 1]]
        cen = counter.from_local(*r.centres(idx).mean(0))
        out['blobs'].append(dict(xy=cen.tolist(), cells=int(len(idx)), top=float(np.max(z[np.isfinite(z)]) - counter.top)))
    return out


BASIN_RIM_BAND_M = 0.07          # the rim zone reaches this far above the counter top: a released object on the rim, below the faucet


def basin_contents(pts: np.ndarray, counter: CounterEst, basin: BasinEst) -> Dict[str, Any]:
    """Zonal occupancy of the basin after a release (rewritten from scratch).
    The parent scored one zone only (points below the counter top) and no phase read
    the result: a release that landed on the rim or across the divider was
    indistinguishable from one that landed in the bowl - the item search masks the
    basin box out, so the object that missed was invisible everywhere in the planner.
    This version scores the basin box in two height zones: 'cells' - points IN the
    bowl (above the floor, clear of the rim plane; a knife or a board lying in the
    basin) and 'rim' - points standing on the rim plane above the counter top (an
    object that missed the bowl and rests on the rim or across the divider). 'top'
    keeps the height of the highest in-bowl point above the floor. The rim zone
    starts above COUNTER_FLAT_M so the counter top itself and the basin walls do
    not count; its upper bound is below the faucet. The sink phase diffs 'cells'
    and 'rim' against the occupancy recorded before the release, so the divider,
    constant, cancels."""
    loc = local_points(pts, counter)
    ba, bb = counter.to_local(basin.centre)
    inside = (np.abs(loc[:, 0] - ba) < 0.5 * basin.size[0] - 0.01) & (np.abs(loc[:, 1] - bb) < 0.5 * basin.size[1] - 0.01)
    z = loc[:, 2]
    bowl = loc[inside & (z > basin.floor + P.BASIN_ITEM_MIN_M) & (z < counter.top - P.BASIN_ITEM_BELOW_TOP_M)]
    rim = loc[inside & (z > counter.top + P.COUNTER_FLAT_M) & (z < counter.top + BASIN_RIM_BAND_M)]
    out: Dict[str, Any] = dict(cells=int(len(bowl)), top=0.0, rim=int(len(rim)))
    if len(bowl):
        out['top'] = float(bowl[:, 2].max() - basin.floor)
    return out
