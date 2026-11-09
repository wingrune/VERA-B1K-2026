"""Object priors of the fair planner of task 42 (data/models.npz, built by planners/fair_tools/build_models.py from the asset collision meshes; see
data/PROVENANCE.json). Properties of the board and parer assets, the same in every instance; no instance pose is stored.

  BOARD_PTS, PARER_PTS     surface samples of the board / the parer in their root frames (z up when they lie on the counter)
  BOARD_Z0, PARER_Z0       height of the root frame above the support they lie on
  BOARD_GRIP, BOARD_AXIS   the strip between the board's handle slot and its end: pinch point (root frame) and closing axis (xy)
  PARER_GRIP, PARER_AXIS   the middle of the parer's handle and the long axis of the knife (blade towards -axis)
  BOWL_INNER               points of the bowl interior (root frame), BASIN_BODY the sink basin box corners (sink frame)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_D = np.load(Path(__file__).resolve().parent / 'data' / 'models.npz')
BOARD_PTS = np.asarray(_D['board_pts'], float)
PARER_PTS = np.asarray(_D['parer_pts'], float)
BOARD_Z0 = -float(np.asarray(_D['board_verts'])[:, 2].min())
PARER_Z0 = -float(np.asarray(_D['parer_verts'])[:, 2].min())
BOARD_GRIP = np.asarray(_D['board_grip'], float)
BOARD_AXIS = np.asarray(_D['board_axis'], float)
PARER_GRIP = np.asarray(_D['parer_grip'], float)
PARER_AXIS = np.asarray(_D['parer_axis'], float)
BOWL_INNER = np.asarray(_D['bowl_inner'], float)
BASIN_BODY = np.asarray(_D['basin_body'], float)
