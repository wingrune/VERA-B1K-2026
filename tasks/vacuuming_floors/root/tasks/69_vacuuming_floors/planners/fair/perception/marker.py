"""The ToggledOn marker of the vacuum handle as an indicator: a red sphere when the vacuum is off and a green one when it is on. Looked for in
a camera image around the position predicted from the vacuum pose; its colour tells the switch state and its centroid (depth back-projected)
the marker position in the robot frame, free of the odometry drift (used to correct the held-vacuum pose)."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from core.skills_lib import rgbd_geometry as G

from .. import params as P


def look(rgb, depth, cam, cam_rel, marker_robot, window_m=P.MARKER_WINDOW_M, green_only=False) -> Optional[Dict[str, Any]]:
    """Colour pixels near the predicted marker.

    Colour pixels near the predicted marker: dict(red, green, state 'on' | 'off' | None, centre (robot frame) or None). None when the
    marker is outside the image or behind something nearer."""
    size, fx = G.CAM_SIZE_FX[cam]
    pix, d = G.project(np.asarray(marker_robot, float)[None], cam, cam_rel)
    if d[0] < 0.08:
        return None
    r, c = int(round(pix[0, 0])), int(round(pix[0, 1]))
    w = int(np.clip(window_m * fx / d[0], 6, 140))
    if not (0 <= r < size and 0 <= c < size):
        return None
    r0, r1, c0, c1 = max(r - w, 0), min(r + w + 1, size), max(c - w, 0), min(c + w + 1, size)
    patch = rgb[r0:r1, c0:c1, :3].astype(int)
    dep = np.asarray(depth, float)[r0:r1, c0:c1]
    near = np.isfinite(dep) & (np.abs(dep - d[0]) < (P.MARKER_DEPTH_TOL_M if window_m <= P.MARKER_WINDOW_M else 0.5))
    red = (patch[..., 0] > P.MARKER_RED[0]) & (patch[..., 1] < P.MARKER_RED[1]) & (patch[..., 2] < P.MARKER_RED[2]) & near
    green = (patch[..., 1] > P.MARKER_GREEN[1]) & (patch[..., 0] < P.MARKER_GREEN[0]) & (patch[..., 2] < P.MARKER_GREEN[2]) & near
    if green_only:
        red = np.zeros_like(red)
    nr, ng = int(red.sum()), int(green.sum())
    state = None
    if ng >= P.MARKER_MIN_PX and ng >= nr:
        state = 'on'
    elif nr >= P.MARKER_MIN_PX and ng < 2:
        state = 'off'
    centre = None
    mask = green if state == 'on' else red if state == 'off' else None
    if mask is not None:
        rows, cols = np.nonzero(mask)
        pts = G.pixels_to_robot(depth, cam, cam_rel, rows + r0, cols + c0)
        if len(pts):
            centre = np.median(pts, axis=0)
    return dict(red=nr, green=ng, state=state, centre=centre, cam=cam)
