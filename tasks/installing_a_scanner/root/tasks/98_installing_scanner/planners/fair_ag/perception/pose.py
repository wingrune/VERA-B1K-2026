"""Frames and pose of the fair planner of task 98: the base pose from the odometry corrected by the heading and wall matching, the desk
frame D, the heading and position corrections from the walls.
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from core.skills_lib import rgbd_geometry as G

from .. import params as P
from .cloud import rot2, self_keep

class PoseMixin:
    def pose(self) -> np.ndarray:
        """Base pose (x, y, yaw) in the odometry frame: the integrated odometry corrected by the map matching."""
        r = self.odo.pose
        return np.array([*(r[:2] + self.corr[:2]), r[2] + self.corr[2]])

    def freeze_walls(self) -> Dict[str, int]:
        """End of the first scan turn: the recorded wall planes become the reference of the position correction."""
        if self.walls.frozen:
            return {}
        return self.walls.freeze()

    def base_d(self) -> np.ndarray:
        """Base pose (x, y, yaw) in the desk frame."""
        r = self.pose()
        return np.array([*(rot2(-self.psi) @ r[:2]), G.wrap(r[2] - self.psi)])

    def to_d(self, xyz: np.ndarray) -> np.ndarray:
        out = np.array(xyz, float, copy=True)
        out[..., :2] = out[..., :2] @ rot2(-self.psi).T
        return out

    def from_d(self, xy: np.ndarray) -> np.ndarray:
        return np.asarray(xy, float)[..., :2] @ rot2(self.psi).T

    def _heading(self, obs: Dict[str, Any], q: np.ndarray, cam_rel: np.ndarray) -> None:
        """Correct the heading of the odometry by the walls and furniture: they run along the axes of the desk frame, so the direction of the
        vertical surfaces in the head view gives the heading modulo a quarter turn, which the odometry (a few degrees off) resolves."""
        if not self.locked or self.freeze_pose:
            return
        grid = G.backproject_grid(G.as_numpy(obs[G.depth_key('head')]), 'head', cam_rel, stride=P.CLOUD_STRIDE)
        flat = grid.reshape(-1, 3)
        valid = np.flatnonzero(np.isfinite(flat[:, 0]))
        flat[valid[~self_keep(flat[valid], q)]] = np.nan                          # the robot's own arms are no wall
        flat[np.hypot(flat[:, 0], flat[:, 1]) > P.HEADING_RANGE_M] = np.nan
        fit = G.wall_heading(grid, min_count=P.HEADING_MIN_POINTS)
        if fit is None or fit[1] < P.HEADING_MIN_CONFIDENCE:
            self.heading_obs = dict(ok=False, conf=None if fit is None else round(fit[1], 2))
            return
        theta_r, conf = fit
        yaw_d = G.wrap(-theta_r)                                                   # wall normals lie on the D axes: yaw_D = -theta_robot (mod quarter turn)
        err = G.wrap(4 * (yaw_d - self.base_d()[2])) / 4                          # difference to the estimate, resolved to the nearest quarter turn
        self.corr[2] += P.HEADING_GAIN * float(np.clip(err, -P.HEADING_MAX_STEP, P.HEADING_MAX_STEP))
        self.heading_obs = dict(ok=True, conf=round(conf, 2), err=round(err, 3))
        self._wall_position(grid)

    def _wall_position(self, grid: np.ndarray) -> None:
        """Record the wall planes during the first scan turn, then correct the position of the base by their shift (drift-free)."""
        pts, normals = G.vertical_surface_samples(grid)
        if len(pts) < P.WALL_MIN_VIEW_POINTS:
            return
        pose = self.base_d()
        c, s = math.cos(pose[2]), math.sin(pose[2])
        rot = np.array([[c, -s], [s, c]])
        pd = pts[:, :2] @ rot.T + pose[:2]
        nd = normals @ rot.T
        if not self.walls.frozen:
            self.walls.record(pd, nd)
            return
        shift = self.walls.observe(pd, nd)
        self.wall_obs = dict(shift=[None if v is None else round(v, 3) for v in shift])
        for k, v in enumerate(shift):
            if v is not None:
                step = -P.WALL_GAIN * float(np.clip(v, -P.WALL_MAX_STEP_M, P.WALL_MAX_STEP_M))
                d = np.zeros(2)
                d[k] = step                                                         # the correction is in the desk frame: back to the odometry frame
                self.corr[:2] += rot2(self.psi) @ d
