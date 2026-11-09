"""Dust detection of the fair planner of task 69: the floor particles are small dark flakes (about 2 x 2.5 cm) lying on the floor, seen in the RGB
of a camera as dark connected blobs whose depth puts them at floor level. Dark things off the floor (the robot, the vacuum, furniture) are
told apart by their depth height, their physical size and the share of black around them."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from scipy import ndimage

from core.skills_lib import rgbd_geometry as G

from .. import params as P


def dust_blobs(rgb, depth, cam, cam_rel, floor_z, ignore=None) -> List[Dict[str, Any]]:
    """Dark flakes on the floor seen by one camera.

    Dark flakes on the floor seen by one camera: robot-frame centres (x, y, z) with their size and darkness.
    rgb (H, W, 3) uint8, depth (H, W) planar metres. `ignore(points_robot) -> bool mask` marks points that are not dust (the robot itself)."""
    size, fx = G.CAM_SIZE_FX[cam]
    lum = rgb[..., :3].astype(np.float32).max(axis=2)
    dark = lum < P.DUST_LUM_MAX
    if not dark.any():
        return []
    black = lum < P.DUST_BLACK_LUM           # the robot and the vacuum: large black regions (a flake's own edge pixels are small)
    if black.any():
        bl, bn = ndimage.label(black, structure=np.ones((3, 3), int))
        big = np.isin(bl, 1 + np.flatnonzero(ndimage.sum(black, bl, np.arange(1, bn + 1)) >= P.DUST_BLACK_MIN_PX))
        near_black = ndimage.binary_dilation(big, iterations=P.DUST_BLACK_PX) if big.any() else big
    else:
        near_black = black
    lab, n = ndimage.label(dark, structure=np.ones((3, 3), int))
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    area = ndimage.sum(dark, lab, idx)
    sl = ndimage.find_objects(lab)
    out = []
    for k, (a, s) in enumerate(zip(area, sl)):
        if a < P.DUST_MIN_PX or a > P.DUST_MAX_PX:
            continue
        rows, cols = np.nonzero(lab[s] == k + 1)
        rows, cols = rows + s[0].start, cols + s[1].start
        d = depth[rows, cols]
        d = d[np.isfinite(d) & (d > G.DEPTH_VALID_M[0]) & (d < P.DUST_RANGE_M)]
        if len(d) < max(2, 0.5 * a):
            continue
        dm = float(np.median(d))
        ext = np.array([rows.max() - rows.min() + 1, cols.max() - cols.min() + 1]) * dm / fx
        if ext.max() > P.DUST_EXTENT_M[1] or ext.max() < P.DUST_EXTENT_M[0]:
            continue
        if near_black[rows, cols].mean() > P.DUST_BLACK_SHARE:
            continue
        v = lum[rows, cols]
        if not (P.DUST_LUM_MEAN[0] <= float(v.mean()) <= P.DUST_LUM_MEAN[1]) or float(np.mean(np.abs(v - np.median(v)) < P.DUST_FLAT_LUM)) < P.DUST_FLAT_SHARE:
            continue                       # a flake is a flat dark-grey patch: not black, not mixed black and white
        r0, r1, c0, c1 = max(s[0].start - 4, 0), min(s[0].stop + 4, lum.shape[0]), max(s[1].start - 4, 0), min(s[1].stop + 4, lum.shape[1])
        ring = lum[r0:r1, c0:c1][(lab[r0:r1, c0:c1] != k + 1)]
        if len(ring) and float(np.mean(ring > P.DUST_RING_LUM)) < P.DUST_RING_SHARE:
            continue                       # the surroundings are not bare floor (the foot of a chair, the edge of an object)
        pts = G.pixels_to_robot(depth, cam, cam_rel, rows, cols)
        if not len(pts):
            continue
        c = pts.mean(0)
        if abs(c[2] - floor_z) > P.DUST_FLOOR_BAND_M:
            continue
        if ignore is not None and not ignore(c[None])[0]:
            continue
        out.append(dict(centre=c, px=int(a), extent=float(ext.max()), lum=float(lum[rows, cols].mean()), depth=dm, cam=cam))
    return out
