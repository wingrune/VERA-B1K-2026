"""Detection and pose of the vacuum from the recent cloud (odometry frame, heights in the robot frame): a tall thin body with cloud points all
through a height band and its top near the handle height; each candidate is confirmed by fitting the asset model (yaw grid + planar ICP on
the floor, one GPU batch, fitkern) and gated by the inlier share."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import ndimage

from .. import fitkern
from .. import models as MD
from .. import params as P
from ..world import VacuumEst

MODEL = MD.PTS[::max(1, len(MD.PTS) // P.FIT_MODEL_PTS)]


def vacuum_candidates(pts: np.ndarray, floor_z: float) -> List[Dict[str, Any]]:
    """Tall thin bodies in the cloud.

    Tall thin bodies in the cloud: connected groups of 3 cm cells holding points of the height band, with a horizontal extent no larger
    than the vacuum's, whose points cover the band's heights and reach the handle height."""
    h = pts[:, 2] - floor_z
    band = pts[(h > P.VAC_BAND_M[0]) & (h < P.VAC_TOP_M[1])]
    if len(band) < P.VAC_MIN_POINTS:
        return []
    lo = band[:, :2].min(0)
    idx = np.floor((band[:, :2] - lo) / P.VAC_CELL_M).astype(int)
    shape = idx.max(0) + 1
    occ = np.zeros(shape, bool)
    occ[idx[:, 0], idx[:, 1]] = True
    lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
    point_lab = lab[idx[:, 0], idx[:, 1]]
    out = []
    for k in range(1, n + 1):
        sel = band[point_lab == k]
        if len(sel) < P.VAC_MIN_POINTS:
            continue
        ext = sel[:, :2].max(0) - sel[:, :2].min(0)
        if ext.max() > P.VAC_MAX_EXTENT_M:
            continue
        hh = sel[:, 2] - floor_z
        if hh.max() < P.VAC_TOP_M[0] or hh.min() > P.VAC_BAND_M[0] + 0.25:
            continue
        # the points cover the band: every 20 cm slice has some
        cover = np.histogram(hh, bins=np.arange(P.VAC_BAND_M[0], P.VAC_TOP_M[0] + 0.01, 0.2))[0]
        if np.any(cover == 0):
            continue
        centre = np.r_[np.median(sel[:, :2], 0), floor_z]
        out.append(dict(centre=centre, n=len(sel), extent=ext, top=float(hh.max())))
    return out


def _subsample(pts: np.ndarray, n: int) -> np.ndarray:
    if len(pts) <= n:
        return pts
    return pts[np.random.default_rng(0).choice(len(pts), n, replace=False)]


def fit_vacuum(pts, floor_z, centre_xy, step, min_inlier=None, roi=P.VAC_ROI_M, hint=None, yaw_deg=None) -> Optional[VacuumEst]:
    """Fit the vacuum model standing on the floor to the cloud...

    Fit the vacuum model standing on the floor to the cloud around centre_xy; None when the best fit fails the inlier gate. With a hint
    (xy and yaw of an earlier estimate of the model frame) only yaws within HINT_YAW_RAD of it are tried, started from its position."""
    h = pts[:, 2] - floor_z
    sel = pts[(np.linalg.norm(pts[:, :2] - centre_xy, axis=1) < roi) & (h > 0.03) & (h < P.VAC_HEIGHT_M + 0.04)]
    cl = _subsample(sel, P.FIT_MAX_CLOUD)
    if len(cl) < P.VAC_MIN_POINTS:
        return None
    z0 = floor_z + MD.FLOOR_Z
    mxy = MODEL[:, :2].mean(0)
    cxy = np.asarray(centre_xy, float)
    if hint is not None:
        span = P.HINT_YAW_DEG if yaw_deg is None else yaw_deg
        yaws = hint['yaw'] + np.radians(np.arange(-span, span + 1e-6, min(P.FIT_YAW_STEP_DEG, max(span / 2., 1.))))
        xy0 = np.tile(np.asarray(hint['xy'], float), (len(yaws), 1))
    else:
        yaws = np.radians(np.arange(0., 360., P.FIT_YAW_STEP_DEG))
        xy0 = np.stack([cxy - np.array([math.cos(a) * mxy[0] - math.sin(a) * mxy[1], math.sin(a) * mxy[0] + math.cos(a) * mxy[1]]) for a in yaws])
    poses, d = fitkern.planar_icp(cl, MODEL, z0, yaws, xy0, P.FIT_ITERS, P.FIT_TRIM_M, P.FIT_BACKEND)
    cost = (np.minimum(d, P.FIT_TRIM_M) ** 2).mean(1)
    inl = (d < P.FIT_INLIER_M).mean(1)
    b = int(np.argmin(cost))
    if inl[b] < (P.VAC_MIN_INLIER if min_inlier is None else min_inlier):
        return None
    x, y, yaw = poses[b]
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.array([[c, -s, 0., x], [s, c, 0., y], [0., 0., 1., z0], [0., 0., 0., 1.]])
    return VacuumEst(T=T, cost=float(cost[b]), inlier=float(inl[b]), n_cloud=len(sel), seen_step=step)
