"""Fit a known 3D object model to an observed point cloud. Scale search and the legacy prior builder are not here.

``object_priors/`` and ``build_object_template.py`` (PLAN, previous tree) are not in this repo. Pass a numpy
model to ``fit_template``. ``fit_rect`` is the known-size rectangle used when only the footprint is known
(the same placement rule as a partial top-down fit).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np


def fit_rect(xy: np.ndarray, yaw: float, size: Sequence[float], prior_centre: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """Centre and coverage in [0, 1] of a known-size rectangle (long, short) at yaw covering xy."""
    xy = np.asarray(xy, float)
    size = (float(size[0]), float(size[1]))
    ex, ey = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
    uv = np.stack([xy @ ex, xy @ ey], 1)
    lo, hi = np.percentile(uv, 1.5, axis=0), np.percentile(uv, 98.5, axis=0)
    pu = None if prior_centre is None else np.array([np.asarray(prior_centre, float) @ ex, np.asarray(prior_centre, float) @ ey])
    c, cover = np.zeros(2), 1.0
    for k in (0, 1):
        span = hi[k] - lo[k]
        mid = (hi[k] + lo[k]) / 2
        if span >= size[k] * 0.93 or pu is None:
            c[k] = mid
        else:
            c[k] = float(np.clip(pu[k], hi[k] - size[k] / 2, lo[k] + size[k] / 2))
        cover = min(cover, span / size[k])
    return c[0] * ex + c[1] * ey, float(min(cover, 1.0))


def _sub(pts: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(pts) <= n:
        return pts
    return pts[rng.choice(len(pts), n, replace=False)]


def _yaw_rot(yaw: float, pts: np.ndarray) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    out = np.array(pts, float, copy=True)
    x, y = pts[:, 0], pts[:, 1]
    out[:, 0] = c * x - s * y
    out[:, 1] = s * x + c * y
    return out


def fit_template(cloud, model, yaw_step_deg: float = 15.0, max_points: int = 250) -> dict:
    """Best yaw about z and translation of model (N,3) onto cloud (M,3). No scale search."""
    rng = np.random.default_rng(0)
    cloud = _sub(np.asarray(cloud, float).reshape(-1, 3), max_points, rng)
    model = _sub(np.asarray(model, float).reshape(-1, 3), max_points, rng)
    if len(cloud) < 3 or len(model) < 3:
        raise ValueError("fit_template needs at least 3 cloud points and 3 model points")
    mc = model.mean(0)
    centred = model - mc
    cc = cloud.mean(0)
    best = None
    step = max(float(yaw_step_deg), 1.0)
    yaw = 0.0
    while yaw < 360.0 - 1e-6:
        rot = _yaw_rot(math.radians(yaw), centred)
        moved = rot + cc
        dz = float(np.median(cloud[:, 2]) - np.median(moved[:, 2]))
        moved = moved + np.array([0.0, 0.0, dz])
        dist = np.linalg.norm(cloud[:, None, :] - moved[None, :, :], axis=-1).min(1).mean()
        centre = np.array([cc[0], cc[1], cc[2] + dz])
        if best is None or dist < best[0]:
            best = (float(dist), centre, math.radians(yaw))
        yaw += step
    dist, centre, yaw_rad = best
    return {"centre": centre, "yaw": float(yaw_rad), "residual": dist, "n_cloud": int(len(cloud))}


def load_object_prior(name: str):
    """Not implemented: object_priors/ and build_object_template.py are not in this repo."""
    raise NotImplementedError(
        f"No stored prior named {name!r}. object_priors/ and build_object_template.py (scale and uint8 fixes) "
        "are not in this repo. Build an (N, 3) model offline and pass it to fit_template."
    )
