"""Object prior of the fair planner of task 69 (data/models.npz, built by planners/fair_tools/build_models.py from the asset collision mesh; see
data/PROVENANCE.json). A property of the vacuum asset, the same in every instance; no instance pose is stored. Frozen.

  VERTS, PTS        collision vertices / surface samples of the vacuum base_link in its own frame (z up when it stands)
  MARKER            the ToggledOn marker centre in that frame (inside the top handle loop)
  CUBE_CENTRE, CUBE_ROT, CUBE_SCALE   the ParticleRemover projection cube: centre, rotation (columns = the cube axes in the body frame) and
                    full extents (long axis along body y)
  FLOOR_Z           height of the body frame origin above the floor when the vacuum stands on it
  HANDLE_LO, HANDLE_HI   box of the top handle loop in the body frame
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_D = np.load(Path(__file__).resolve().parent / 'data' / 'models.npz')
VERTS = np.asarray(_D['verts'], float)
PTS = np.asarray(_D['pts'], float)
MARKER = np.asarray(_D['marker_body'], float)
CUBE_CENTRE = np.asarray(_D['cube_centre_body'], float)
CUBE_ROT = np.asarray(_D['cube_rot_body'], float)
CUBE_SCALE = np.asarray(_D['cube_scale'], float)
FLOOR_Z = float(_D['base_link_above_floor'])
HANDLE_LO = np.asarray(_D['handle_lo'], float)
HANDLE_HI = np.asarray(_D['handle_hi'], float)
