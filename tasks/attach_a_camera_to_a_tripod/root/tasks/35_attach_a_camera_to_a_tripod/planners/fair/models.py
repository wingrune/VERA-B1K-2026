"""Object priors of the fair planner of task 35 (data/models.npz, built by planners/fair_tools/build_models.py from the asset collision meshes; see
data/PROVENANCE.json). Properties of the camera and tripod assets, the same in every instance; no instance pose is stored.

  CAMERA_VERTS, CAMERA_PTS   collision vertices / surface samples of the camera in its root frame (z up when it rests on the floor)
  CAMERA_M                   4x4 attachment frame M (screw hole under the camera) in the camera root frame
  TRIPOD_VERTS, TRIPOD_PTS   the same for the tripod (upright)
  TRIPOD_F                   4x4 attachment frame F (top of the tripod head) in the tripod root frame
  CAMERA_FLOOR, TRIPOD_FLOOR height of the root frame above the floor when the object stands on it (minus the lowest vertex)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_D = np.load(Path(__file__).resolve().parent / 'data' / 'models.npz')
CAMERA_VERTS = np.asarray(_D['camera_verts'], float)
CAMERA_PTS = np.asarray(_D['camera_pts'], float)
CAMERA_M = np.asarray(_D['camera_attach_T'], float)
TRIPOD_VERTS = np.asarray(_D['tripod_verts'], float)
TRIPOD_PTS = np.asarray(_D['tripod_pts'], float)
TRIPOD_F = np.asarray(_D['tripod_attach_T'], float)
CAMERA_FLOOR = -float(CAMERA_VERTS[:, 2].min())
TRIPOD_FLOOR = -float(TRIPOD_VERTS[:, 2].min())
TRIPOD_FEET = TRIPOD_VERTS[TRIPOD_VERTS[:, 2] < TRIPOD_VERTS[:, 2].min() + 0.06]
