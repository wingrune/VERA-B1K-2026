"""Vegetables and baskets on the floor from coloured clouds: points low over the floor classified by colour (corn yellow, onion pink-purple,
bok choy / leek / broccoli green), grouped into blobs per class on a fine top-down raster; a blob gives the vegetable's box, the long axis
of its footprint and its top. Baskets are the wicker-coloured points round the vegetables. Group labels: corn -> 2, onion -> 0; a green
vegetable is bok choy (group 0) in the basket with the corn and a leek or broccoli (group 1) in the basket with the onions (BDDL init)."""
from __future__ import annotations

from typing import List

import numpy as np
from scipy import ndimage

from core.skills_lib import rgbd_geometry as RG

from .. import params as P
from ..world import BasketEst, VegEst


def classes(col: np.ndarray) -> np.ndarray:
    """Colour class per point: 0 none, 1 corn, 2 onion, 3 green, 4 wicker."""
    r, g, b = (col[:, k].astype(float) for k in range(3))
    out = np.zeros(len(col), np.int8)
    out[(r >= P.CORN_R_MIN) & (g >= P.CORN_G_FRAC[0] * r) & (g <= P.CORN_G_FRAC[1] * r) & (b <= P.CORN_B_FRAC * r)] = 1
    out[(r >= P.ONION_R_MIN) & (g <= P.ONION_G_MAX * r) & (np.abs(g - b) <= P.ONION_GB_MAX * r) & (b >= 0.3 * r)] = 2
    out[(g >= P.GREEN_G_MIN) & (g >= P.GREEN_OVER_R * r) & (g >= P.GREEN_OVER_B * b)] = 3
    wick = (r >= P.WICKER_R_MIN) & (g >= P.WICKER_G_FRAC[0] * r) & (g <= P.WICKER_G_FRAC[1] * r) & (b >= P.WICKER_B_FRAC[0] * r) & (b <= P.WICKER_B_FRAC[1] * r)
    out[wick & (out == 0)] = 4
    return out


KIND = {1: 'corn', 2: 'onion', 3: 'green'}


def blobs(pts: np.ndarray, res: float, min_points: int):
    """Connected components of the points on a top-down raster: list of point index arrays."""
    if len(pts) < min_points:
        return []
    lo = pts[:, :2].min(0) - 2 * res
    idx = np.floor((pts[:, :2] - lo) / res).astype(int)
    grid = np.zeros(idx.max(0) + 3, bool)
    grid[idx[:, 0], idx[:, 1]] = True
    lab, n = ndimage.label(ndimage.binary_closing(grid), structure=np.ones((3, 3)))
    out = []
    for k in range(1, n + 1):
        m = np.flatnonzero(lab[idx[:, 0], idx[:, 1]] == k)
        if len(m) >= min_points:
            out.append(m)
    return out


def find_vegs(pts: np.ndarray, col: np.ndarray, step: int) -> List[VegEst]:
    low = (pts[:, 2] > P.VEG_Z_M[0]) & (pts[:, 2] < P.VEG_Z_M[1])
    cls = classes(col)
    out = []
    for c, kind in KIND.items():
        sel = np.flatnonzero(low & (cls == c))
        for m in blobs(pts[sel], P.VEG_RES_M, P.VEG_MIN_POINTS):
            p = pts[sel[m]]
            centre, yaw, ext = RG.min_area_rect(p[:, :2])
            if ext[1] > ext[0]:
                yaw, ext = yaw + np.pi / 2, ext[::-1]
            if ext[0] > P.VEG_MAX_LEN_M or ext[1] < P.VEG_MIN_WIDTH_M:     # slivers: cabinet plinth edges, a chair leg (native 302 scan)
                continue
            top = float(np.percentile(p[:, 2], 95))
            out.append(VegEst(kind=kind, centre=np.r_[centre, 0.5 * (top + max(0., float(p[:, 2].min())))], top=top,
                              axis=np.array([np.cos(yaw), np.sin(yaw)]), length=float(ext[0]), width=float(ext[1]), n_points=len(p),
                              seen_step=step, pts=p))
    return out


def merge(tracks: List[VegEst], found: List[VegEst], step: int) -> None:
    """Replace the tracks near each new sighting of the same kind (Perception.expire drops the tracks that views stop finding)."""
    for f in found:
        near = [k for k, t in enumerate(tracks) if t.kind == f.kind and np.linalg.norm(t.centre[:2] - f.centre[:2]) < P.VEG_MATCH_M]
        if near:
            k = near[0]
            f.group, f.basket = tracks[k].group, tracks[k].basket
            tracks[k] = f
        else:
            tracks.append(f)


def update_baskets(baskets: List[BasketEst], pts: np.ndarray, col: np.ndarray, step: int) -> None:
    """Wicker blobs of a basket's size low over the floor."""
    cls = classes(col)
    sel = np.flatnonzero((cls == 4) & (pts[:, 2] > 0.02) & (pts[:, 2] < P.BASKET_TOP_M))
    for m in blobs(pts[sel], P.BASKET_RES_M, P.BASKET_MIN_POINTS):
        p = pts[sel[m]]
        centre, yaw, ext = RG.min_area_rect(p[:, :2])
        if not (P.BASKET_EXT_M[0] <= ext.min() and ext.max() <= P.BASKET_EXT_M[1]):
            continue
        rim = float(np.percentile(p[:, 2], 97))
        if not P.BASKET_RIM_BAND_M[0] <= rim <= P.BASKET_RIM_BAND_M[1]:     # a wooden thing 0.39 m high by the dining table of 302
            continue
        near = [b for b in baskets if np.linalg.norm(b.centre - centre) < P.BASKET_MATCH_M]
        if near:
            b = near[0]
            b.centre, b.yaw, b.size, b.rim, b.n, b.seen_step = centre, yaw, ext, rim, b.n + 1, step
        else:
            baskets.append(BasketEst(centre=centre, yaw=yaw, size=ext, rim=rim, seen_step=step))


def label_groups(vegs: List[VegEst], baskets: List[BasketEst]) -> None:
    """Basket of each vegetable (nearest basket centre within BASKET_REACH_M), kinds per basket, and the group of each vegetable."""
    for b in baskets:
        b.kinds = {}
    for v in vegs:
        d = [float(np.linalg.norm(b.centre - v.centre[:2])) for b in baskets]
        v.basket = int(np.argmin(d)) if d and min(d) < P.BASKET_REACH_M else None
        if v.basket is not None and v.basket < len(baskets):
            baskets[v.basket].kinds[v.kind] = baskets[v.basket].kinds.get(v.kind, 0) + 1
            v.rim = baskets[v.basket].rim
    for v in vegs:
        if v.basket is None:           # every vegetable starts in a basket (BDDL); a yellow thing by the counter of 301 read as corn
            v.group = None
        elif v.kind == 'corn':
            v.group = 2
        elif v.kind == 'onion':
            v.group = 0
        elif v.basket is not None and v.basket < len(baskets):
            k = baskets[v.basket].kinds
            if k.get('corn', 0) > k.get('onion', 0):
                v.group = 0
            elif k.get('onion', 0) > k.get('corn', 0):
                v.group = 1
