"""Localisation of the fair planner against the offline scene map: head points of a full turn in place (odometry frame, the band of walls and
furniture) are matched to the map's likelihood field. Global search: every heading (2 deg), all translations at once by FFT correlation, the
robot cell required free in the map; the best poses are refined and kept as competing hypotheses with their scores. Local refinement: a small
window around the current transform with the latest view, accepted only when it scores better by a margin."""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage, signal

from .scene_map import SceneMap

BAND = (0.10, 1.40)          # map-frame heights of the points that are matched (as in the map builder)
CELL = 0.05
GLOBAL_STEP_DEG = 2.0
TOP_K = 6
MIN_CLEARANCE_M = 0.22       # the robot centre of a hypothesis is at least this far from any occupied map cell


def rot(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def unique_cells(xy: np.ndarray, res: float = CELL) -> np.ndarray:
    """One point per occupied cell (cell centres)."""
    if len(xy) == 0:
        return xy.reshape(0, 2)
    c = np.unique(np.floor(xy / res).astype(np.int64), axis=0)
    return (c + 0.5) * res


def score(m: SceneMap, pts: np.ndarray, T: np.ndarray) -> float:
    """Mean likelihood of the odometry-frame points under the transform T = (tx, ty, theta) odometry -> map."""
    w = pts @ rot(T[2]).T + T[:2]
    return float(m.sample(m.likelihood, w).mean()) if len(w) else 0.


def compose(T: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Map pose of an odometry pose."""
    xy = rot(T[2]) @ pose[:2] + T[:2]
    return np.array([xy[0], xy[1], math.atan2(math.sin(pose[2] + T[2]), math.cos(pose[2] + T[2]))])


def refine(m: SceneMap, pts: np.ndarray, T: np.ndarray, span_xy: float, span_th: float, steps: int = 3) -> Tuple[np.ndarray, float]:
    """Coordinate-wise grid refinement around T, shrinking the window each round."""
    best, bs = np.asarray(T, float).copy(), score(m, pts, T)
    dxy, dth = span_xy, span_th
    for _ in range(steps):
        for th in np.linspace(-dth, dth, 7):
            for dx in np.linspace(-dxy, dxy, 7):
                for dy in np.linspace(-dxy, dxy, 7):
                    cand = best + np.array([dx, dy, th])
                    s = score(m, pts, cand)
                    if s > bs:
                        bs, best = s, cand
        dxy, dth = dxy / 3, dth / 3
    return best, bs


def global_search(m: SceneMap, pts: np.ndarray, robot_odom: np.ndarray) -> List[Dict]:
    """Hypotheses [{T, score}] best first for odometry-frame points (N, 2) seen around the robot at odometry pose robot_odom."""
    pts = unique_cells(pts)
    if len(pts) < 50:
        return []
    lik = m.likelihood
    free = (m.clearance >= MIN_CLEARANCE_M) & ndimage.binary_dilation(m.seen, iterations=6)
    cands = []
    for deg in np.arange(0., 360., GLOBAL_STEP_DEG):
        th = math.radians(deg)
        r = rot(th)
        w = (pts - robot_odom[:2]) @ r.T            # points relative to the robot, rotated into the map heading
        lo = np.floor(w.min(0) / m.res).astype(int)
        idx = np.floor(w / m.res).astype(int) - lo
        img = np.zeros(idx.max(0) + 1)
        np.add.at(img, (idx[:, 0], idx[:, 1]), 1.)
        corr = signal.fftconvolve(lik, img[::-1, ::-1], mode='full')
        # corr[a, b] = sum img[i, j] * lik[a - (H-1) + i, b - (W-1) + j]: robot cell = (a - (H-1) - lo0, b - (W-1) - lo1)
        H, W = img.shape
        a0, b0 = H - 1 + lo[0], W - 1 + lo[1]
        sub = np.full(m.shape, -1.)
        ai, bi = np.arange(m.shape[0]) + a0, np.arange(m.shape[1]) + b0
        oka = (ai >= 0) & (ai < corr.shape[0])
        okb = (bi >= 0) & (bi < corr.shape[1])
        sub[np.ix_(oka, okb)] = corr[np.ix_(ai[oka], bi[okb])]
        sub[~free] = -1.
        for _ in range(2):
            k = int(np.argmax(sub))
            i, j = np.unravel_index(k, sub.shape)
            if sub[i, j] <= 0:
                break
            robot_map = m.origin + (np.array([i, j]) + 0.5) * m.res
            cands.append((float(sub[i, j]) / len(pts), th, robot_map))
            sub[max(0, i - 8):i + 9, max(0, j - 8):j + 9] = -1.
    cands.sort(key=lambda c: -c[0])
    out: List[Dict] = []
    for s, th, robot_map in cands[:4 * TOP_K]:
        T = np.r_[robot_map - rot(th) @ robot_odom[:2], th]
        T, sc = refine(m, pts, T, 0.06, math.radians(2.0))
        if any(np.linalg.norm(T[:2] - h['T'][:2]) < 0.3 and abs(math.remainder(T[2] - h['T'][2], 2 * math.pi)) < 0.2 for h in out):
            continue
        out.append(dict(T=T, score=sc))
        if len(out) >= TOP_K:
            break
    out.sort(key=lambda h: -h['score'])
    return out


def band_points(pts_map_z: np.ndarray) -> np.ndarray:
    z = pts_map_z[:, 2]
    return pts_map_z[(z > BAND[0]) & (z < BAND[1])][:, :2]


def smooth_free(m: SceneMap) -> np.ndarray:
    return ndimage.binary_opening(m.seen & ~m.occ)
