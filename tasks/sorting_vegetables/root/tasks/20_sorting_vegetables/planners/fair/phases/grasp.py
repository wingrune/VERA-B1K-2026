"""Top pinches of a vegetable from its perceived points (as the oracle pinches on the collision hull, here on the blob seen from above):
a slab across the local long axis whose top run fits the gripper; the fingers close across it at the run's mid-height (its visible height
capped by the kind's thickness). Options whose finger spots or hand edges are near the basket's walls, or under another vegetable, are
dropped (oracle findings, native 301)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .. import geometry as G
from .. import params as P
from ..world import BasketEst, VegEst, World


def hits_wall(mid: np.ndarray, jaw: np.ndarray, spread: float, b: BasketEst) -> bool:
    """A finger landing spot or an edge of the hand closer than WALL_MARGIN_M to a wall of the basket (its perceived oriented footprint)."""
    across = np.cross(G.UP, G.horiz(jaw))
    c, s = np.cos(b.yaw), np.sin(b.yaw)
    for spot in (mid + spread * G.horiz(jaw), mid - spread * G.horiz(jaw), mid + P.HAND_HALF_M * across, mid - P.HAND_HALF_M * across):
        d = spot[:2] - b.centre
        local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
        if np.min(0.5 * b.size - np.abs(local)) < P.WALL_MARGIN_M:
            return True
    return False


def columns_blocked(mid: np.ndarray, jaw: np.ndarray, spread: float, top: float, others: List[np.ndarray]) -> bool:
    """Points of another vegetable over this one's top on the columns of the pinch point, both finger spots and both hand edges (the
    descent stops on it: a leek across an onion stopped the fingers 11-21 cm high, oracle native 301)."""
    if not others:
        return False
    pts = np.concatenate(others)
    pts = pts[pts[:, 2] > top + 0.01]
    if not len(pts):
        return False
    across = np.cross(G.UP, G.horiz(jaw))
    for xy in (mid, mid + spread * G.horiz(jaw), mid - spread * G.horiz(jaw), mid + P.HAND_HALF_M * across, mid - P.HAND_HALF_M * across):
        if np.any(np.linalg.norm(pts[:, :2] - xy[:2], axis=1) < P.COLUMN_R_M):
            return True
    return False


def fingers_blocked(mid: np.ndarray, jaw: np.ndarray, spread: float, cloud: Optional[np.ndarray]) -> bool:
    """Anything seen at a finger's landing spot higher than FINGER_CLEAR_M over the pinch height (a neighbour in the pile, or the
    vegetable itself when it is wider than its top suggested): the descent would stop on it (t20f10: two descents stopped 9 cm short)."""
    if cloud is None or not len(cloud):
        return False
    j = G.horiz(jaw)
    for spot in (mid + spread * j, mid - spread * j):
        near = cloud[np.linalg.norm(cloud[:, :2] - spot[:2], axis=1) < P.FINGER_COL_R_M]
        if len(near) and float(near[:, 2].max()) > mid[2] + P.FINGER_CLEAR_M:
            return True
    return False


def kind_thick(v: VegEst) -> float:
    """Height of the vegetable lying down (assets): a long thin green blob is a leek."""
    if v.kind == 'green' and v.length >= P.LEEK_MIN_LEN_M and v.width <= P.LEEK_MAX_WIDTH_M:
        return P.LEEK_THICK_M
    return P.KIND_THICK_M.get(v.kind, 0.06)


def pinch_options(v: VegEst, w: Optional[World] = None, tally: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
    """Pinch options of the vegetable blob, cheapest first: dict(site, mid, pre, rotation, width, axis, top, thick, cost). Touching
    vegetables of one colour make one blob (three corns of 302), so the pinches are found locally: candidate centres on a PINCH_GRID_M
    raster of the blob's upper points; at each, the local long axis (points within PINCH_LOCAL_M), the slab across it, the run of the
    slab's points around the centre that stay within RUN_DEPTH_M of the centre's top (a ridge; the valley between two corns ends it), whose
    width must fit the gripper. Higher vegetables of a pile first (nothing lies on them)."""
    pts = v.pts
    if pts is None or len(pts) < 8:
        return []
    basket = w.baskets[v.basket] if w is not None and v.basket is not None and v.basket < len(w.baskets) else None
    others = [] if w is None else [o.pts for o in w.vegs if o is not v and o.pts is not None
                                   and np.linalg.norm(o.xy - v.xy) < 0.4 and np.linalg.norm(o.xy - v.xy) > 0.02]
    top_all = float(np.percentile(pts[:, 2], 98))
    cells = np.floor(pts[:, :2] / P.PINCH_GRID_M).astype(int)
    seeds = {}
    for k, cell in enumerate(map(tuple, cells)):             # the highest point of each raster cell
        if cell not in seeds or pts[k, 2] > pts[seeds[cell], 2]:
            seeds[cell] = k
    out = []
    tally = {} if tally is None else tally
    count = lambda why: tally.__setitem__(why, tally.get(why, 0) + 1)   # noqa: E731
    for k in seeds.values():
        c0 = pts[k]
        near = pts[np.linalg.norm(pts[:, :2] - c0[:2], axis=1) < P.PINCH_LOCAL_M]
        if len(near) < 6:
            continue
        d = near[:, :2] - near[:, :2].mean(0)
        _, vecs = np.linalg.eigh(d.T @ d)
        dirs = [vecs[:, 1], vecs[:, 0]]                         # across the long axis first, then across the short one
        if v.kind in P.ROUND_KINDS:                            # a round one also diagonally: by a basket wall the jaw must run along it
            dirs += [(vecs[:, 1] + vecs[:, 0]) / np.sqrt(2.), (vecs[:, 1] - vecs[:, 0]) / np.sqrt(2.)]
        for u2 in dirs:
            u, a = np.r_[u2, 0.], np.r_[-u2[1], u2[0], 0.]
            rel = near - c0
            slab = near[np.abs(rel @ u) < P.SLAB_HALF_M]
            if len(slab) < 4:
                continue
            t, z = (slab - c0) @ a, slab[:, 2]
            col = np.abs(t) < P.COLUMN_R_M
            if not col.any():
                continue
            top = float(z[col].max())
            if top < z.max() - P.KIND_VALLEY_M.get(v.kind, P.VALLEY_M):   # a valley: the centre lies between two vegetables
                count('valley')
                continue
            order = np.argsort(t)
            ts, keep = t[order], z[order] >= top - P.KIND_RUN_DEPTH_M.get(v.kind, P.RUN_DEPTH_M)
            i0 = i1 = int(np.argmin(np.abs(ts)))
            while i0 > 0 and keep[i0 - 1] and ts[i0] - ts[i0 - 1] < P.RUN_GAP_M:
                i0 -= 1
            while i1 < len(ts) - 1 and keep[i1 + 1] and ts[i1 + 1] - ts[i1] < P.RUN_GAP_M:
                i1 += 1
            wa = float(ts[i1] - ts[i0])
            if not P.MIN_RUN_M <= wa <= P.KIND_MAX_WIDTH_M.get(v.kind, P.VEG_MAX_WIDTH_M):
                count('width')
                continue
            centre = c0 + a * 0.5 * (ts[i0] + ts[i1])
            thick = kind_thick(v)          # the kind's height lying down: the visible run (2-3 cm) put the fingertips over the equator of
            # onions and corns and they slid over the top (closed empty, native t20f10); the oracle closes under the mid-height
            mid = np.r_[centre[:2], max(top - 0.5 * thick - P.PINCH_DEEPER_M, P.FLOOR_CLEAR_M)]
            spread = P.GRASP_SPREAD_M
            why = ('wall' if basket is not None and hits_wall(mid, a, spread, basket) else 'covered' if columns_blocked(mid, a, spread, top, others)
                   else 'fingers' if fingers_blocked(mid, a, spread, None if w is None else w.cloud) else None)
            if why:
                count(why)
                continue
            pre_z = max(mid[2] + P.VEG_BACK_M, (v.rim or 0.) + P.RIM_CLEAR_M)
            out.append(dict(site='ridge', mid=mid, pre=np.r_[mid[:2], pre_z], rotation=G.jaw_rotation(G.DOWN, a), width=wa, axis=u[:2],
                            top=top, thick=thick, cost=2.0 * max(0., wa - 0.06) + P.LOW_COST * (top_all - top)))
            break
    out.sort(key=lambda o: o['cost'])
    kept: List[Dict[str, Any]] = []
    for o in out:                                              # one option per spot
        if all(np.linalg.norm(o['mid'][:2] - k['mid'][:2]) > P.PINCH_APART_M for k in kept):
            kept.append(o)
    return kept


def pinch_from(v: VegEst, w: Optional[World] = None) -> Optional[Dict[str, Any]]:
    opts = pinch_options(v, w)
    return opts[0] if opts else None
