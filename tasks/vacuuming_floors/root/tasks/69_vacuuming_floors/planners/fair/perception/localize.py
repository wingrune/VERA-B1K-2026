"""Drift correction of the odometry pose against the obstacle map, while the base stands.

The pose drifts 0.1-0.3 m: a small search over shifts and turns matches the head's obstacle points to the blocked cells.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage

from .. import params as P


def _score_map(perc: Any) -> np.ndarray:
    occ = perc.anchor
    d = ndimage.distance_transform_edt(~occ) * P.OBST_RES_M
    return 0.6 * np.exp(-(d / P.LOC_REACH_M) ** 2) + 0.4 * np.exp(-(d / (3 * P.LOC_REACH_M)) ** 2)      # a sharp peak on the cell and a wide skirt to pull in


def _score(sm: np.ndarray, xy: np.ndarray) -> float:
    c = np.clip(np.floor((xy + P.OBST_HALF_M) / P.OBST_RES_M).astype(int), 0, sm.shape[0] - 1)
    return float(sm[c[:, 0], c[:, 1]].sum())


def correct(perc: Any, head_robot: np.ndarray) -> None:
    """Move the pose when the head points fit the map better."""
    h = head_robot[:, 2] - perc.floor_z
    r = np.hypot(head_robot[:, 0], head_robot[:, 1])
    keep = (h > P.LOC_BAND_M[0]) & (h < P.LOC_BAND_M[1]) & (r > P.OBST_SELF_R_M) & (r < P.LOC_RANGE_M)
    pts = head_robot[keep, :2]
    if perc.anchor is None or len(pts) < P.LOC_MIN_POINTS or int(perc.anchor.sum()) < P.LOC_MIN_CELLS:
        return
    pts = pts[:: max(1, len(pts) // P.LOC_MAX_POINTS)]
    perc.counts['loc_runs'] = perc.counts.get('loc_runs', 0) + 1
    sm = _score_map(perc)
    b = perc.pose()

    def world(dx: float, dy: float, dyaw: float) -> np.ndarray:
        c, s = math.cos(b[2] + dyaw), math.sin(b[2] + dyaw)
        return np.stack([b[0] + dx + c * pts[:, 0] - s * pts[:, 1], b[1] + dy + s * pts[:, 0] + c * pts[:, 1]], 1)

    base_score = _score(sm, world(0., 0., 0.))
    n = float(len(pts))
    best = (base_score, 0., 0., 0.)
    for span, step, yspan, ystep in ((P.LOC_SHIFT_M, 0.05, P.LOC_TURN_RAD, 0.03), (0.04, 0.0125, 0.03, 0.01)):      # coarse, then fine around the coarse best
        cx, cy, cyaw = best[1:]
        for dyaw in np.arange(-yspan, yspan + 1e-9, ystep):
            for dx in np.arange(-span, span + 1e-9, step):
                for dy in np.arange(-span, span + 1e-9, step):
                    ex, ey, eyw = cx + dx, cy + dy, cyaw + dyaw
                    pen = P.LOC_PENALTY * n * ((ex * ex + ey * ey) / P.LOC_SHIFT_M ** 2 + (eyw / P.LOC_TURN_RAD) ** 2)
                    s = _score(sm, world(ex, ey, eyw)) - pen
                    if s > best[0] + 1e-9:
                        best = (s, ex, ey, eyw)
    gain = (best[0] - base_score) / n
    d = np.zeros(3)
    if gain >= P.LOC_MIN_GAIN and base_score / n <= P.LOC_GOOD_FIT:
        d = P.LOC_GAIN * np.array(best[1:])
        perc.odo.pose = perc.odo.pose + d
        perc.counts['loc_applied'] = perc.counts.get('loc_applied', 0) + 1
        perc.counts['loc_shift_mm'] = perc.counts.get('loc_shift_mm', 0) + int(1000 * math.hypot(d[0], d[1]))
        perc.loc_log.append(dict(step=perc.step, d=[round(float(v), 3) for v in best[1:]], gain=round(float(gain), 3), fit0=round(base_score / n, 2), n=int(n)))
    if _score(sm, world(*d)) / n >= P.LOC_VERIFIED:      # this view fits the map: the cells it hits (at the checked pose) join the reference
        w = world(*d)
        c = np.clip(np.floor((w + P.OBST_HALF_M) / P.OBST_RES_M).astype(int), 0, sm.shape[0] - 1)
        perc.anchor[c[:, 0], c[:, 1]] = True
