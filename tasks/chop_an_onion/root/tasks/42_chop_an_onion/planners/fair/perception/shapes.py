"""Pose of the board and the parer on the counter by fitting their asset models (yaw grid + planar ICP on the counter, one GPU batch, fitkern)
to the cloud of the detected component. The yaw hypothesis with the least cloud-to-model distance wins; the model's asymmetry (the board's handle,
the knife's blade and handle) breaks the 180 degree ambiguity."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .. import fitkern
from .. import models as MD
from .. import params as P
from ..world import CounterEst, ItemEst
from .scene import local_points

MODEL = {'board': (MD.BOARD_PTS, MD.BOARD_Z0, MD.BOARD_GRIP, MD.BOARD_AXIS), 'parer': (MD.PARER_PTS, MD.PARER_Z0, MD.PARER_GRIP, MD.PARER_AXIS)}


def _slot_hits(cloud, pose, z_min) -> int:
    """Points in the slot box.
    Cloud points on the board's top (above z_min) that fall into the model's slot box at the fitted pose."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    d = cloud[:, :2] - np.array([x, y])
    mx, my = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
    (x0, x1), (y0, y1) = P.SLOT_BOX_M
    return int(np.sum((mx > x0) & (mx < x1) & (np.abs(my) < y1) & (my > y0) & (cloud[:, 2] > z_min)))


def fit_shape(item: ItemEst, pts: np.ndarray, counter: CounterEst, step: int) -> Optional[ItemEst]:
    """Item pose from the model fit.
    Item with yaw, centre, grip point and closing axis from the model fit; None when the fit is poor (inlier share below FIT_MIN_INLIER)."""
    model, z0, grip, axis = MODEL[item.role]
    loc = local_points(pts, counter)
    c_loc = counter.to_local(item.centre[:2])
    near = np.linalg.norm(loc[:, :2] - c_loc, axis=1) < 0.5 * float(item.size[0]) + P.FIT_ROI_PAD_M
    h = loc[:, 2] - counter.top
    cloud = loc[near & (h > P.FIT_Z_BAND_M[0]) & (h < P.FIT_Z_BAND_M[1])]
    if len(cloud) < P.FIT_MIN_POINTS:
        return None
    if len(cloud) > P.FIT_MAX_CLOUD:
        cloud = cloud[np.random.default_rng(0).choice(len(cloud), P.FIT_MAX_CLOUD, replace=False)]
    u_yaw = math.atan2(counter.u[1], counter.u[0])
    yaws = np.radians(np.arange(0., 360., P.FIT_YAW_STEP_DEG))
    mxy = model[:, :2].mean(0)
    cxy = cloud[:, :2].mean(0)
    xy0 = np.stack([cxy - np.array([math.cos(a) * mxy[0] - math.sin(a) * mxy[1], math.sin(a) * mxy[0] + math.cos(a) * mxy[1]]) for a in yaws])
    poses, d = fitkern.planar_icp(cloud, model, counter.top + z0, yaws, xy0, P.FIT_ITERS, P.FIT_TRIM_M, P.FIT_BACKEND)
    cost = (np.minimum(d, P.FIT_TRIM_M) ** 2).mean(1)
    inl = (d < P.FIT_INLIER_M).mean(1)
    b = int(np.argmin(cost))
    if item.role == 'board':             # the handle slot is a hole: the right end has no board points inside it; the flipped pose competes
        fp, fd = fitkern.planar_icp(cloud, model, counter.top + z0, [poses[b, 2] + math.pi], poses[b:b + 1, :2], 0, P.FIT_TRIM_M, P.FIT_BACKEND)
        flip = np.array([fp[0, 0], fp[0, 1], poses[b, 2] + math.pi])
        hits = (_slot_hits(cloud, poses[b], counter.top + P.SLOT_MIN_H_M), _slot_hits(cloud, flip, counter.top + P.SLOT_MIN_H_M))
        if hits[1] + P.SLOT_HIT_MARGIN < hits[0]:
            poses = np.vstack([poses, flip[None]])
            cost = np.r_[cost, (np.minimum(fd[0], P.FIT_TRIM_M) ** 2).mean()]
            inl = np.r_[inl, (fd[0] < P.FIT_INLIER_M).mean()]
            b = len(poses) - 1
    if inl[b] < P.FIT_MIN_INLIER:
        return None
    x, y, yaw = poses[b]
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    root_xy = counter.from_local(x, y)
    yaw_odom = yaw + u_yaw
    ca, sa = math.cos(yaw_odom), math.sin(yaw_odom)
    Ro = np.array([[ca, -sa], [sa, ca]])
    lo, hi = model.min(0), model.max(0)
    mid_xy = root_xy + Ro @ (0.5 * (lo[:2] + hi[:2]))
    grip_xy = root_xy + Ro @ grip[:2]
    axis_xy = Ro @ axis
    out = ItemEst(role=item.role, centre=np.array([mid_xy[0], mid_xy[1], counter.top + 0.5 * (hi[2] - lo[2])]), top=counter.top + float(hi[2] + z0),
                  size=np.array([hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]]), yaw=float(yaw_odom), n_points=item.n_points, fit=float(inl[b]),
                  seen_step=step, grip=np.array([grip_xy[0], grip_xy[1], counter.top + z0 + float(grip[2])]), axis=axis_xy,
                  extra=dict(cost=float(cost[b]), second=float(np.sort(cost)[1])))
    return out
