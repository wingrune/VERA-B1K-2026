"""Detection and pose of the camera and the tripod from the recent cloud (odometry frame, heights in the robot frame).

Camera: a compact low body on the floor with nothing above it. Tripod: a compact head at the tripod head height with a column under it and
nothing above it. Each candidate is confirmed by fitting the asset model (yaw grid + planar ICP on the floor, one GPU batch, fitkern) and
ranking the hypotheses by the cloud-to-model distance plus the share of the model that the cameras look through. The tripod legs are
three-fold symmetric: the distinct yaw minima are kept as ranked hypotheses (the pan handle separates them when it is seen).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import ndimage

from .. import fitkern
from .. import models as MD
from .. import params as P
from ..world import ObjectEst


def _components(xy: np.ndarray, cell: float, mask_fn=None):
    """Connected groups (8-neighbour) of the occupied cells of the points xy; returns (labels per point, number, cell index per point)."""
    lo = xy.min(0)
    idx = np.floor((xy - lo) / cell).astype(int)
    shape = idx.max(0) + 1
    occ = np.zeros(shape, bool)
    occ[idx[:, 0], idx[:, 1]] = True
    if mask_fn is not None:
        occ &= mask_fn(lo, shape)
    labels, n = ndimage.label(occ, structure=np.ones((3, 3)))
    return labels[idx[:, 0], idx[:, 1]], n, lo


def camera_candidates(pts: np.ndarray, floor_z: float) -> List[Dict[str, Any]]:
    h = pts[:, 2] - floor_z
    low = pts[(h > P.CAM_BAND_M[0]) & (h < P.CAM_BAND_M[1])]
    high = pts[(h > P.CAM_CLEAR_BAND_M[0]) & (h < P.CAM_CLEAR_BAND_M[1])]
    if len(low) < P.CAM_MIN_POINTS:
        return []

    def clear(lo, shape):
        blocked = np.zeros(shape, bool)
        if len(high):
            k = np.floor((high[:, :2] - lo) / P.CAM_CELL_M).astype(int)
            ok = np.all((k >= 0) & (k < shape), 1)
            blocked[k[ok, 0], k[ok, 1]] = True
            blocked = ndimage.binary_dilation(blocked, iterations=2)
        return ~blocked

    lab, n, _ = _components(low[:, :2], P.CAM_CELL_M, clear)
    out = []
    for k in range(1, n + 1):
        sel = low[lab == k]
        if len(sel) < P.CAM_MIN_POINTS:
            continue
        ext = sel[:, :2].max(0) - sel[:, :2].min(0)
        if ext.max() > P.CAM_EXTENT_M[1] or ext.max() < P.CAM_EXTENT_M[0]:
            continue
        centre = np.r_[(sel[:, :2].max(0) + sel[:, :2].min(0)) / 2, floor_z]
        out.append(dict(centre=centre, n=len(sel), extent=ext))
    return out


def tripod_candidates(pts: np.ndarray, floor_z: float) -> List[Dict[str, Any]]:
    h = pts[:, 2] - floor_z
    head = pts[(h > P.TRIPOD_HEAD_Z_M[0]) & (h < P.TRIPOD_HEAD_Z_M[1])]
    if len(head) < 8:
        return []
    column = pts[(h > P.TRI_COLUMN_BAND_M[0]) & (h < P.TRI_COLUMN_BAND_M[1])]
    above = pts[(h > P.TRI_ABOVE_BAND_M[0]) & (h < P.TRI_ABOVE_BAND_M[1])]
    lab, n, _ = _components(head[:, :2], P.TRI_CELL_M)
    out = []
    for k in range(1, n + 1):
        sel = head[lab == k]
        if len(sel) < 8:
            continue
        ext = sel[:, :2].max(0) - sel[:, :2].min(0)
        if ext.max() > P.TRI_HEAD_EXTENT_M:
            continue
        c = np.median(sel[:, :2], 0)
        n_col = int(np.sum(np.linalg.norm(column[:, :2] - c, axis=1) < P.TRI_COLUMN_R_M)) if len(column) else 0
        n_above = int(np.sum(np.linalg.norm(above[:, :2] - c, axis=1) < P.TRI_ABOVE_R_M)) if len(above) else 0
        if n_col < P.TRI_COLUMN_MIN or n_above > 3:
            continue
        out.append(dict(centre=np.r_[c, floor_z], n=len(sel), n_column=n_col))
    return out


def _subsample(pts: np.ndarray, n: int) -> np.ndarray:
    if len(pts) <= n:
        return pts
    return pts[np.random.default_rng(0).choice(len(pts), n, replace=False)]


def fit(cloud: np.ndarray, model: np.ndarray, z0: float, centre_xy: np.ndarray, views: list, step: int, min_inlier: float,
        n_keep: int = 1, see_weight: Optional[float] = None) -> Optional[ObjectEst]:
    """Fit `model` standing on the floor (origin at height z0) to `cloud` around centre_xy; None when the best fit fails the inlier gate.
    The estimate keeps up to n_keep distinct yaw minima (20 degrees apart), best first."""
    cl = _subsample(cloud, P.FIT_MAX_CLOUD)
    if len(cl) < P.CAM_MIN_POINTS:
        return None
    yaws = np.radians(np.arange(0., 360., P.FIT_YAW_STEP_DEG))
    mxy = model[:, :2].mean(0)
    cxy = cl[:, :2].mean(0)
    xy0 = np.stack([cxy - np.array([math.cos(a) * mxy[0] - math.sin(a) * mxy[1], math.sin(a) * mxy[0] + math.cos(a) * mxy[1]]) for a in yaws])
    poses, d = fitkern.planar_icp(cl, model, z0, yaws, xy0, P.FIT_ITERS, P.FIT_TRIM_M, P.FIT_BACKEND)
    probe = model[::max(1, len(model) // 1500)]
    see_weight = P.SEE_THROUGH_WEIGHT if see_weight is None else see_weight
    use = views and len(probe) and see_weight > 0
    shown = fitkern.see_through(probe, poses, z0, views, P.SEE_THROUGH_TOL_M, P.FIT_BACKEND) if use else np.zeros(len(poses))
    cost = (np.minimum(d, P.FIT_TRIM_M) ** 2).mean(1) + see_weight * shown
    inl = (d < P.FIT_INLIER_M).mean(1)
    order = np.argsort(cost)
    keep: List[int] = []
    for i in order:
        if all(abs(math.atan2(math.sin(poses[i, 2] - poses[j, 2]), math.cos(poses[i, 2] - poses[j, 2]))) > math.radians(20) for j in keep):
            keep.append(int(i))
        if len(keep) >= n_keep:
            break
    b = keep[0]
    if inl[b] < min_inlier:
        return None
    x, y, yaw = poses[b]
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.array([[c, -s, 0., x], [s, c, 0., y], [0., 0., 1., z0], [0., 0., 0., 1.]])
    return ObjectEst(T=T, cost=float(cost[b]), inlier=float(inl[b]), n_cloud=len(cloud), seen_step=step,
                     yaws=[float(poses[i, 2]) for i in keep], costs=[float(cost[i]) for i in keep])


def fit_camera(pts: np.ndarray, floor_z: float, cand: Dict[str, Any], views: list, step: int) -> Optional[ObjectEst]:
    c = cand['centre']
    h = pts[:, 2] - floor_z
    sel = pts[(np.linalg.norm(pts[:, :2] - c[:2], axis=1) < P.CAM_ROI_M) & (h > 0.004) & (h < P.CAMERA_HEIGHT_M + 0.03)]
    if len(sel) and np.percentile(sel[:, 2] - floor_z, 90) < 0.045:     # flat patch (rug, paper): not a camera body
        return None
    return fit(sel, MD.CAMERA_PTS, floor_z + MD.CAMERA_FLOOR, c[:2], views, step, P.CAM_MIN_INLIER, see_weight=P.CAM_SEE_THROUGH_WEIGHT)


HEAD_PTS = MD.TRIPOD_PTS[MD.TRIPOD_PTS[:, 2] > MD.TRIPOD_F[2, 3] - P.TRI_HEAD_MODEL_DEPTH_M]


def fit_tripod(pts: np.ndarray, floor_z: float, cand: Dict[str, Any], views: list, step: int) -> Optional[ObjectEst]:
    """Whole-model fit (yaw hypotheses from the legs and the handle), then the position refined on the head alone with the yaw kept: the F
    frame sits on the head, and the legs' collision shapes pull the whole fit off by about 2 cm."""
    c = cand['centre']
    h = pts[:, 2] - floor_z
    sel = pts[(np.linalg.norm(pts[:, :2] - c[:2], axis=1) < P.TRI_ROI_M) & (h > 0.03) & (h < P.TRIPOD_HEAD_Z_M[1] + 0.03)]
    est = fit(sel, MD.TRIPOD_PTS, floor_z + MD.TRIPOD_FLOOR, c[:2], views, step, P.TRI_MIN_INLIER, n_keep=3)
    if est is None:
        return None
    F = est.T @ MD.TRIPOD_F
    start = est.T[:2, 3].copy()
    for _ in range(2):      # refine twice: the second pass re-centres the head region on the refined position
        F = est.T @ MD.TRIPOD_F
        head = pts[(np.linalg.norm(pts[:, :2] - F[:2, 3], axis=1) < P.TRI_HEAD_ROI_M) & (h > P.TRI_HEAD_BAND_M[0]) & (h < P.TRI_HEAD_BAND_M[1])]
        if len(head) < 20:
            break
        poses, _ = fitkern.planar_icp(_subsample(head, P.FIT_MAX_CLOUD), HEAD_PTS, est.T[2, 3], [est.yaws[0]], est.T[None, :2, 3], P.FIT_ITERS,
                                      P.TRI_HEAD_TRIM_M, P.FIT_BACKEND, rotate=False)
        if np.linalg.norm(poses[0, :2] - start) < 0.08:
            est.T[:2, 3] = poses[0, :2]
        else:
            break
    return est
