"""Perception of the fair planner of task 98: the native observation (3 RGB-D cameras + proprio) to a World in the desk frame.

Input is only the observation dict. The base pose comes from integrating the measured base velocity (odometry frame: the base at the first
control); the desk top, found as the dominant horizontal surface at table height, then defines the working frame D (odometry rotated so the desk
axes are the axes), in which everything is reported. Layers kept over time: obstacle hits around the base (5 cm), desk-top hits (2 cm). Objects
on the desk (scanner: tall, laptop: flat) are measured from the current frame only. The button is found as the template point of the scanner;
saturated-red pixels (the marker, seen in RGB-D) decide which end of the scanner carries it. The marker colour is never used to decide
that the button was pressed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from core.skills_lib import r1pro_kinematics as K
from core.skills_lib import rgbd_geometry as G

from .. import params as P
from ..walls import WallMap
from ..world import Box, ScannerEst, World
from .button import ButtonMixin
from .cloud import CAMS, CloudMixin
from .colour import ColourMixin
from .laptop import LaptopMixin
from .navgrid import NavGridMixin
from .objects import ObjectsMixin
from .pose import PoseMixin
from .scanner import ScannerMixin
from .table import TableMixin

class Perception(CloudMixin, PoseMixin, TableMixin, ObjectsMixin, ScannerMixin, LaptopMixin, ColourMixin, ButtonMixin, NavGridMixin):
    """Parts: cloud (clouds), pose (frames, heading, walls), table (layers, desk), objects/scanner/laptop, colour (red blob), button, navgrid."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        n = int(2 * P.OBST_HALF_M / P.OBST_RES_M)
        m = int(2 * P.OBST_HALF_M / P.TOP_RES_M)
        self.obst = np.zeros((n, n), np.uint8)
        self.tall = np.zeros((n, n), np.uint8)
        self.floor_seen = np.zeros((n, n), np.uint8)       # head-camera hits on the floor: cells known to be free ground
        self.visited = np.zeros((n, n), bool)             # cells the base has been in
        self.freeze_pose = False                          # no pose corrections while an arm works at the scanner: the pose estimate must not jump
        self.allow_partial = False                        # track a partly hidden scanner (set while the scanner is being pushed)
        self.held: Optional[Dict[str, Any]] = None        # scanner held by the assisted grasp (side, grasp transform, box): its points are not the scene
        self.top_hits = np.zeros((m, m), np.uint8)
        self.odo = G.Odometry()
        self.lift = 0.0                                 # how far the base is raised above the floor (m), from the floor seen by the head camera
        self.bumps = np.zeros((n, n), bool)             # places where the base climbed onto something: hard obstacles from then on
        self.tilt = (0., 0.)                            # floor slope of the robot frame (a, b) of z = a x + b y + c
        self.corr = np.zeros(3)                         # correction of the odometry pose by what the cameras see (only the heading is corrected)
        self.heading_obs: Dict[str, Any] = {}
        self.walls = WallMap()
        self.wall_obs: Dict[str, Any] = {}
        self.step = 0
        self.psi, self.locked = 0., False
        self.top_z: Optional[float] = None
        self.z_hist = np.zeros(int(round((P.DESK_TOP_RANGE_M[1] - P.DESK_TOP_RANGE_M[0]) / 0.01)))
        self.desk_box: Optional[Box] = None
        self.desk_area = 0.
        self.scanner: Optional[ScannerEst] = None
        self.laptop_box: Optional[Box] = None
        self.marker_blob: Optional[np.ndarray] = None     # last red blob centre (D frame), diagnostic and front decision
        self.marker_blob_step = -10 ** 9
        self.counts = dict(updates=0, desk_frames=0, scanner_frames=0, laptop_frames=0, blob_frames=0)
        self.proprio: Optional[np.ndarray] = None
        self.cache_version = 0
        self.last_clusters: List[tuple] = []

    def observe(self, obs: Dict[str, Any], step: int, perceive: Optional[bool] = None, base_override: Optional[np.ndarray] = None) -> None:
        """Integrate the odometry every control and run the full perception every PERCEIVE_EVERY controls (or when `perceive`)."""
        proprio = np.asarray(G.as_numpy(obs[G.PROPRIO_KEY]), float).reshape(-1)
        self.proprio = proprio
        if base_override is None:
            self.odo.update(proprio[:3])
        else:
            self.odo.pose = np.asarray(base_override, float).copy()
        self.step = step
        if perceive is None:
            perceive = step % P.PERCEIVE_EVERY == 0
        if perceive:
            self._perceive(obs, proprio)

    def _perceive(self, obs: Dict[str, Any], proprio: np.ndarray) -> None:
        self.counts['updates'] += 1
        q = K.q_from_proprio(proprio)
        cam_rel = np.asarray(G.as_numpy(obs[G.CAM_POSES_KEY]), float).reshape(-1)
        clouds = self._clouds(obs, q, cam_rel)
        self._lift(clouds['head'][0])
        clouds = self._level(clouds)
        self._heading(obs, q, cam_rel)
        self.corr[2] = float(np.clip(self.corr[2], -0.4, 0.4))   # wall snaps can lock onto the wrong wall axis: a heading correction this large is not drift
        pose = self.pose()
        clouds = {c: (G.robot_to_odom(p, pose), col) for c, (p, col) in clouds.items()}
        head = clouds['head'][0]
        self._update_layers(head)
        self._update_desk(head)
        if self.top_z is not None and self.desk_box is not None:
            pts = np.concatenate([clouds[c][0] for c in CAMS])
            self._update_objects(self.to_d(pts))
            self._update_marker(obs, q, cam_rel)

    def world(self) -> World:
        return World(step=self.step, base=self.base_d(), scanner=self.scanner, marker=self.marker_estimate(), laptop_box=self.laptop_box,
                     table_box=self.desk_box)

    def est(self) -> Dict[str, Any]:
        """Estimates in the odometry frame, for the harness log (the harness compares them to the truth)."""
        out: Dict[str, Any] = dict(base=self.pose()[:2].tolist(), base_yaw=float(self.pose()[2]), heading=dict(self.heading_obs), lift=round(float(self.lift), 3), tilt=[round(float(v), 4) for v in self.tilt], corr=[round(float(v), 3) for v in self.corr], psi=round(self.psi, 3), locked=self.locked, top_z=self.top_z,
                                   desk_area=round(self.desk_area, 2), counts=dict(self.counts))
        s = self.scanner
        if s is not None:
            out['scanner'] = self.from_d(s.centre).tolist()
            out['marker'] = self.from_d(self.marker_estimate()[:2]).tolist()
            out['scanner_yaw'] = round(G.wrap(s.yaw + self.psi), 3)
            out['front_known'] = s.front_known
        if self.laptop_box is not None:
            out['laptop'] = self.from_d(self.laptop_box.centre[:2]).tolist()
        if self.marker_blob is not None:
            out['marker_blob'] = self.from_d(self.marker_blob[:2]).tolist()
        return out
