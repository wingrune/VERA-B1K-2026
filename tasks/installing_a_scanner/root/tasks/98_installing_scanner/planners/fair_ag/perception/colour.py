"""The red marker of the fair planner of task 98: saturated-red pixels of the three cameras, lifted to 3D by the depth, give the button position.
The marker colour is only a position, never a sign that the button was pressed.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.skills_lib import rgbd_geometry as G

from .. import params as P
from .cloud import CAMS

class ColourMixin:
    def _update_marker(self, obs: Dict[str, Any], q: np.ndarray, cam_rel: np.ndarray) -> None:
        for cam in CAMS:
            rgb = np.asarray(G.as_numpy(obs[G.rgb_key(cam)]))[..., :3].astype(np.int16)
            r = rgb[..., 0]
            red = (r >= P.RED_MIN_R) & (rgb[..., 1] < P.RED_MAX_GB_FRAC * r) & (rgb[..., 2] < P.RED_MAX_GB_FRAC * r)
            rows, cols = np.nonzero(red)
            if len(rows) < P.RED_MIN_PIXELS:
                continue
            pts = G.pixels_to_robot(G.as_numpy(obs[G.depth_key(cam)]), cam, cam_rel, rows, cols)
            if len(pts) < P.RED_MIN_PIXELS:
                continue
            c_robot = np.median(pts, 0)
            cam_pos = G.cam_pose(cam_rel, cam)[:3, 3]
            view = (c_robot - cam_pos) / max(np.linalg.norm(c_robot - cam_pos), 1e-6)
            c_robot = c_robot + 0.009 * view            # the centroid of the visible cap lies in front of the sphere centre
            centre = self.to_d(G.robot_to_odom(c_robot[None], self.pose()))[0]
            if self.top_z is not None and not (self.top_z - 0.02 < centre[2] < self.top_z + 0.2):
                continue
            if self.scanner is not None and np.linalg.norm(centre[:2] - self.scanner.centre) > 0.3:
                continue
            self.marker_blob, self.marker_blob_step = centre, self.step
            self.counts['blob_frames'] += 1
        self._decide_front()
