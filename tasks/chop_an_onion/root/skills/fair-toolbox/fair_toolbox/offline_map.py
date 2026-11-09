"""Offline occupancy map built from training episodes, queried at test time from RGB-D only.

Cells are occupied, seen-free, or unseen. Localisation keeps competing base hypotheses. A point in a
seen-free cell is an unexpected object where the map is empty. Points past the map border are not a miss.
The episode-dump reader is task-specific and is not implemented here.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _cell(xy: np.ndarray, origin, res: float) -> np.ndarray:
    return np.floor((np.asarray(xy, float) - np.asarray(origin, float)) / float(res)).astype(int)


def _raster(clouds: Sequence[np.ndarray], origin, shape, res: float) -> np.ndarray:
    acc = np.zeros(tuple(shape), np.int32)
    h, w = acc.shape
    for xy in clouds:
        pts = np.asarray(xy, float).reshape(-1, 2)
        if not len(pts):
            continue
        c = _cell(pts, origin, res)
        ok = (c[:, 0] >= 0) & (c[:, 1] >= 0) & (c[:, 0] < h) & (c[:, 1] < w)
        if not ok.any():
            continue
        flat = c[ok, 0] * w + c[ok, 1]
        np.add.at(acc.reshape(-1), flat, 1)
    return acc


def from_points(obstacle_xy, floor_xy, origin, shape, res: float = 0.05, min_hits: int = 4):
    """(occupied, seen) bool grids from world-frame training clouds. origin is the corner of cell (0, 0)."""
    hits = _raster(obstacle_xy, origin, shape, res)
    floor = _raster(floor_xy, origin, shape, res)
    occupied = hits >= int(min_hits)
    seen = (floor > 0) | occupied
    return occupied, seen


def free_cells(occupied, seen) -> np.ndarray:
    """(N, 2) indices of cells that were seen and are not occupied."""
    return np.argwhere(np.asarray(seen, bool) & ~np.asarray(occupied, bool))


def unexpected(points_xy, occupied, seen, origin, res: float) -> np.ndarray:
    """Mask of points that land in a seen, unoccupied cell (an object where the map is empty)."""
    occ = np.asarray(occupied, bool)
    sn = np.asarray(seen, bool)
    pts = np.asarray(points_xy, float).reshape(-1, 2)
    c = _cell(pts, origin, res)
    h, w = occ.shape
    inside = (c[:, 0] >= 0) & (c[:, 1] >= 0) & (c[:, 0] < h) & (c[:, 1] < w)
    out = np.zeros(len(pts), bool)
    if inside.any():
        cc = c[inside]
        out[inside] = sn[cc[:, 0], cc[:, 1]] & ~occ[cc[:, 0], cc[:, 1]]
    return out


def localise(points_robot, hypotheses, occupied, seen, origin, res: float, keep: float = 0.85) -> dict:
    """Rank (x, y, yaw) hypotheses by in-map points on occupied cells. Outside points are ignored, not a miss."""
    pts = np.asarray(points_robot, float).reshape(-1, 3)
    hyps = np.atleast_2d(np.asarray(hypotheses, float))
    occ = np.asarray(occupied, bool)
    h, w = occ.shape
    scores = np.zeros(len(hyps))
    for i, (x, y, yaw) in enumerate(hyps):
        c, s = math.cos(float(yaw)), math.sin(float(yaw))
        world = np.stack([x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]], 1)
        cells = _cell(world, origin, res)
        inside = (cells[:, 0] >= 0) & (cells[:, 1] >= 0) & (cells[:, 0] < h) & (cells[:, 1] < w)
        if not inside.any():
            continue
        cc = cells[inside]
        scores[i] = float(occ[cc[:, 0], cc[:, 1]].mean())
    best = float(scores.max()) if len(scores) else 0.0
    competing = scores >= (best * float(keep) if best > 0 else best)
    order = np.argsort(-scores)
    return {"order": order, "scores": scores, "competing": competing}


def head_slices(depth, cam_rel, bands=((0.10, 0.40), (0.40, 0.80), (0.80, 1.20), (1.20, 1.60)), stride: int = 8):
    """Robot-frame head points split into height bands for map matching."""
    from core.skills_lib import rgbd_geometry as geom
    pts, _pix, _col = geom.backproject(depth, "head", cam_rel, stride=stride)
    out = []
    for lo, hi in bands:
        if not len(pts):
            out.append(pts)
            continue
        out.append(pts[(pts[:, 2] >= lo) & (pts[:, 2] < hi)])
    return out


def build_from_episode_dumps(_paths):
    """Not implemented: training-dump layout and offline truth poses are task-specific. Use from_points."""
    raise NotImplementedError(
        "Episode observation dumps and the offline use of harness truth poses differ per task "
        "(a task map builder owns that). Transform training clouds into the map frame and call from_points. "
        "Do not build the map from scene assets or from evaluation seeds."
    )
