"""Pose of the base for the fair planner of task 83: the integrated odometry, its heading corrected by the walls. The rooms are rectangular, so
the vertical surfaces in the head view give the heading modulo a quarter turn; the room axes are fixed in the odometry frame by the first
confident view (the odometry is exact at the start) and every later view pulls the heading back onto them."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from core.skills_lib import rgbd_geometry as G

from .. import params as P
from .cloud import self_keep


class PoseMixin:
    def pose(self) -> np.ndarray:
        """Base pose (x, y, yaw) in the odometry frame: the integrated odometry plus the heading correction."""
        r = self.odo.pose
        return np.array([r[0], r[1], G.wrap(r[2] + self.yaw_corr)])

    def _heading(self, obs: Dict[str, Any], q: np.ndarray, cam_rel: np.ndarray) -> None:
        if self.freeze_pose:
            return
        grid = G.backproject_grid(G.as_numpy(obs[G.depth_key('head')]), 'head', cam_rel, stride=P.CLOUD_STRIDE)
        flat = grid.reshape(-1, 3)
        valid = np.flatnonzero(np.isfinite(flat[:, 0]))
        flat[valid[~self_keep(flat[valid], q)]] = np.nan
        flat[np.hypot(flat[:, 0], flat[:, 1]) > P.HEADING_RANGE_M] = np.nan
        fit = G.wall_heading(grid, min_count=P.HEADING_MIN_POINTS)
        if fit is None or fit[1] < P.HEADING_MIN_CONFIDENCE:
            self.heading_obs = dict(ok=False, conf=None if fit is None else round(fit[1], 2))
            return
        theta_r, conf = fit
        yaw = self.pose()[2]
        if self.room_psi is None:
            self.room_psi = G.wrap(yaw + theta_r)            # a wall normal of the room, in the odometry frame
            self.heading_obs = dict(ok=True, conf=round(conf, 2), err=0.0, psi=round(self.room_psi, 4))
            return
        err = G.wrap(4 * (self.room_psi - theta_r - yaw)) / 4  # wall normals lie on the room axes: yaw + theta = psi (mod a quarter turn)
        self.yaw_corr += P.HEADING_GAIN * float(np.clip(err, -P.HEADING_MAX_STEP, P.HEADING_MAX_STEP))
        self.heading_obs = dict(ok=True, conf=round(conf, 2), err=round(err, 4))

    @property
    def room_axes(self) -> Optional[np.ndarray]:
        """Unit xy of the room axes (2 x 2) in the odometry frame, or None before the first wall view."""
        if self.room_psi is None:
            return None
        c, s = np.cos(self.room_psi), np.sin(self.room_psi)
        return np.array([[c, s], [-s, c]])
