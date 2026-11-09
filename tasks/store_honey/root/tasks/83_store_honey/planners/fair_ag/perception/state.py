"""Perception of the fair planner of task 83: the native observation (3 RGB-D cameras + proprio) to a World in the odometry frame.

Input is only the observation dict. The base pose comes from integrating the measured base velocity (odometry frame: the base at the first
control), its heading pulled onto the room axes seen in the walls. Layers kept over time: obstacle cells around the base, the decaying cabinet-top
layer. The cabinet is found by its top (height band and size of the asset), the drawer face by the vertical surface in front of its top edge.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from core.skills_lib import r1pro_kinematics as K
from core.skills_lib import rgbd_geometry as G

from .. import params as P
from ..world import CabinetEst, JarEst, World
from .cabinet import CabinetMixin
from .cloud import CAMS, CloudMixin, self_keep
from .jar import JarMixin
from .layers import LayersMixin
from .pose import PoseMixin


class Perception(CloudMixin, PoseMixin, LayersMixin, CabinetMixin, JarMixin):
    """Parts: cloud (clouds, lift, tilt), pose (odometry, heading), layers (obstacles), cabinet (top, face), jar (sightings, height layer)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        n = int(2 * P.MAP_HALF_M / P.MAP_RES_M)
        m = int(2 * P.MAP_HALF_M / P.TOP_RES_M)
        self.obst = np.zeros((n, n), np.uint8)
        self.floor_seen = np.zeros((n, n), bool)
        self.visited = np.zeros((n, n), bool)
        self.top = np.zeros((m, m), np.float32)
        self.odo = G.Odometry()
        self.yaw_corr = 0.0
        self.room_psi: Optional[float] = None
        self.heading_obs: Dict[str, Any] = {}
        self.lift = 0.0
        self.tilt = (0., 0.)
        self.freeze_pose = False            # no heading corrections while a hand works at the cabinet
        self.freeze_cabinet = False         # the estimate is kept while the drawer moves (its top would grow the rectangle)
        self.cabinet: Optional[CabinetEst] = None
        self.heights = np.full((int(2 * P.MAP_HALF_M / P.HEIGHT_RES_M),) * 2, np.nan, np.float32)
        self.jar: Optional[JarEst] = None
        self.jar_lock: Optional[JarEst] = None    # strongest fused jar so far; sparse far groups do not replace it
        self.jar_lock_n = 0                       # sightings of the group that set the lock
        self.jar_sightings: list = []
        self.freeze_jar = False             # the jar estimate is kept while a hand is near it (the self filter cuts its points)
        self.jar_dense = False              # a denser head cloud for the jar search
        self.cab_diag: list = []                 # top clusters of the last measurement: centre xy, length, depth, fill
        self.step = 0
        self.proprio: Optional[np.ndarray] = None
        self.counts = dict(updates=0, cabinet_frames=0)

    def observe(self, obs: Dict[str, Any], step: int, perceive: Optional[bool] = None) -> None:
        """Integrate the odometry every control and run the full perception every PERCEIVE_EVERY controls (or when `perceive`)."""
        proprio = np.asarray(G.as_numpy(obs[G.PROPRIO_KEY]), float).reshape(-1)
        self.proprio = proprio
        self._integrate(proprio[:3])
        self.step = step
        if perceive is None:
            perceive = step % P.PERCEIVE_EVERY == 0
        if perceive:
            self._perceive(obs, proprio)

    def _integrate(self, base_vel: np.ndarray) -> None:
        """Odometry step; with ODO_CORRECTED the translation is integrated along the corrected heading (the raw odometry turns it with its
        own drifting yaw, so every heading correction would leave the position on the old track)."""
        xy, yaw0 = self.odo.pose[:2].copy(), float(self.odo.pose[2])
        self.odo.update(base_vel)
        if P.ODO_CORRECTED:
            vx, vy, wz = (self.odo.scale * float(v) for v in base_vel[:3])
            yaw = yaw0 + self.yaw_corr + 0.5 * wz * self.odo.dt
            c, s = np.cos(yaw), np.sin(yaw)
            self.odo.pose[:2] = xy + np.array([c * vx - s * vy, s * vx + c * vy]) * self.odo.dt

    def _perceive(self, obs: Dict[str, Any], proprio: np.ndarray) -> None:
        self.counts['updates'] += 1
        q = K.q_from_proprio(proprio)
        cam_rel = np.asarray(G.as_numpy(obs[G.CAM_POSES_KEY]), float).reshape(-1)
        clouds = self._clouds(obs, q, cam_rel)
        self._lift(clouds['head'])
        clouds = self._level(clouds)
        self._heading(obs, q, cam_rel)
        pose = self.pose()
        odom = {c: G.robot_to_odom(p, pose) for c, p in clouds.items()}
        self._update_layers(odom['head'])
        allpts = np.concatenate([odom[c] for c in CAMS])
        self._update_top(allpts)
        self._update_heights(allpts)
        eye = G.robot_to_odom(G.cam_pose(cam_rel, 'head')[:3, 3][None], pose)[0, :2]
        if self.jar_dense:
            dense, _, _ = G.backproject(G.as_numpy(obs[G.depth_key('head')]), 'head', cam_rel, stride=P.JAR_DENSE_STRIDE)
            a, b = self.tilt
            dense[:, 2] -= a * dense[:, 0] + b * dense[:, 1]
            dense = dense[(np.linalg.norm(dense[:, :2], axis=1) < P.JAR_RANGE_M) & (dense[:, 2] > P.JAR_SUPPORT_MIN_M) & (dense[:, 2] < P.JAR_TOP_MAX_M)]
            dense = dense[self_keep(dense, q)]
            jar_pts = np.concatenate([G.robot_to_odom(dense, pose), odom['left_wrist'], odom['right_wrist']])
        else:
            jar_pts = allpts
        self._update_jar(jar_pts, eye)

    def world(self) -> World:
        return World(step=self.step, base=self.pose(), cabinet=self.cabinet, jar=self.jar)

    def est(self) -> Dict[str, Any]:
        """Estimates in the odometry frame, for the harness log (the harness compares them to the truth)."""
        pose = self.pose()
        out: Dict[str, Any] = dict(base=pose[:2].tolist(), base_yaw=float(pose[2]), heading=dict(self.heading_obs), lift=round(float(self.lift), 3),
                                   tilt=[round(float(v), 4) for v in self.tilt], yaw_corr=round(self.yaw_corr, 4), counts=dict(self.counts))
        c = self.cabinet
        if c is not None:
            out['cabinet'] = c.centre.tolist()
            out['cabinet_f'] = np.round(c.f, 3).tolist()
            out['cabinet_size'] = [round(c.length, 3), round(c.depth, 3), round(c.top_z, 3)]
            out['cabinet_face'] = [round(c.face, 3), c.face_measured, c.complete]
            out['ledges'] = [np.round(c.ledge(k), 3).tolist() for k in (1, -1)]
        j = self.jar
        if j is not None:
            out['jar'] = [round(float(v), 3) for v in (*j.xy, j.support, j.top)]
            out['jar_sightings'] = j.sightings
        return out
