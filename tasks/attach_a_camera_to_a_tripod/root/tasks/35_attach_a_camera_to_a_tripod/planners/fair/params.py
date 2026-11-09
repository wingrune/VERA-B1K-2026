"""Constants of the fair planner of task 35 (mutated separately from the logic). Units: metres, radians, controls (30 Hz).

Heights are measured from the floor seen by the head camera (perception estimates the floor height in the robot frame)."""

# ---- object priors (asset geometry, data/models.npz; see data/PROVENANCE.json) -------------------------------------------------------------
CAMERA_HEIGHT_M = 0.091            # collision box height of the camera resting on the floor
TRIPOD_HEAD_Z_M = (1.10, 1.33)      # height band of the tripod head (F lies at about 1.28 m)

# ---- perception -------------------------------------------------------------------------------------------------------------------------
PERCEIVE_EVERY = 5                 # controls between perception updates
HEAD_STRIDE, WRIST_STRIDE = 3, 4   # pixel strides of the back-projection
CLOUD_RANGE_M = 4.5                # points farther than this (horizontal) from the base are dropped
SELF_RADIUS_M = 0.085              # points this near an arm link segment are the robot's own
FINGER_RADIUS_M = 0.05             # and this near a finger
BODY_RADIUS_M = 0.36               # points this near the base axis (horizontal) are the robot's own
HELD_RADIUS_M = 0.14               # points this near the predicted centre of the held camera are the camera, not the scene
RECENT_UPDATES = 6                 # perception updates kept in the recent cloud (detection and fits)
FLOOR_BAND_M = 0.02                # head points within this of the floor height are floor
OBST_RES_M, OBST_HALF_M = 0.05, 12.0
OBST_BAND_M = (0.06, 1.7)          # heights of the points that block the base
OBST_MIN_HITS = 2
OBST_SELF_R_M = 0.5                # obstacle points nearer than this to the base (horizontal) are the robot's own (torso, bumper)
TORSO_CUT = True                   # cut the torso (torso links and head, TORSO_RADIUS_M) out of the clouds
TORSO_RADIUS_M = 0.17
OBST_FROM_WRISTS = False           # wrist points mark obstacles too (with hanging arms the wrist cameras see the robot's own base)
TRIPOD_IN_GRID = True              # the fitted tripod's footprint (vertices below TRIPOD_GRID_H_M) blocks the base in the route grid
TRIPOD_GRID_H_M = 0.6
MAP_HOLD_PHASES = ('pre_grasp', 'descend', 'close', 'back_off', 'lift', 'back_away', 'tuck', 'to_pre_mount', 'insert', 'retract',
                   'release')       # phases without obstacle updates (the arm moves with the camera in the views)
CLEAR_GRASPED, CLEAR_GRASPED_R_M = True, 0.15   # forget the obstacle hits where the grasped camera lay
UNKNOWN_COST = 1.0                 # extra path cost factor of cells never seen

# camera detection: a compact low body on the floor with nothing above it
CAM_BAND_M = (0.012, 0.115)
CAM_CLEAR_BAND_M = (0.14, 0.7)
CAM_CELL_M = 0.02
CAM_EXTENT_M = (0.05, 0.27)
CAM_MIN_POINTS = 25
CAM_ROI_M = 0.16                   # points this near a candidate centre go to the fit
# tripod detection: a compact head at its height, a column under it, nothing above it
TRI_CELL_M = 0.03
TRI_HEAD_EXTENT_M = 0.36
TRI_COLUMN_BAND_M = (0.55, 1.05)
TRI_COLUMN_R_M = 0.10
TRI_COLUMN_MIN = 6
TRI_ABOVE_BAND_M = (1.36, 1.9)
TRI_ABOVE_R_M = 0.30
TRI_ROI_M = 0.56
TRI_HEAD_MODEL_DEPTH_M = 0.25      # head refinement: model points this far below F and up
TRI_HEAD_ROI_M, TRI_HEAD_BAND_M = 0.30, (0.95, 1.36)
TRI_HEAD_MIN_POINTS, TRI_HEAD_TRIM_M, TRI_HEAD_MAX_SHIFT_M = 40, 0.02, 0.05
# model fits (batched yaw grid + ICP on the GPU)
FIT_BACKEND = 'gpu'
FIT_YAW_STEP_DEG = 10.0
FIT_ITERS = 25
FIT_TRIM_M = 0.03
FIT_INLIER_M = 0.012
FIT_MAX_CLOUD = 500
CAM_MIN_INLIER = 0.65               # true fits 0.78-0.99, wrong objects 0.47-0.51 on the port dumps
TRI_MIN_INLIER = 0.6
SEE_THROUGH_TOL_M = 0.04           # a model point is contradicted when the camera sees this much farther through it
SEE_THROUGH_WEIGHT = 0.03          # cost per contradicted fraction (the cost is a squared distance in m^2)
CAM_SEE_THROUGH_WEIGHT = 0.0       # the camera fit's own weight (0: cloud distance only; on the port dumps 0 gives yaw errors median 2.4, max 14 deg)

# ---- navigation -------------------------------------------------------------------------------------------------------------------------
ROBOT_RADIUS = 0.34
RELAX_RADIUS = 0.28                # every route failed: retry once with this footprint radius (0: no retry; the base's own is 0.28)
GRID_RES = 0.05
NAV_GAIN, NAV_VFAR, NAV_VNEAR, NAV_YAW_GAIN, NAV_WMAX = 0.9, 0.4, 0.15, 0.8, 0.5
NAV_POS_TOL, NAV_YAW_TOL = 0.04, 0.06
NEAR_POS_TOL, NEAR_YAW_TOL, NEAR_STALL_STEPS = 0.10, 0.20, 60
NAV_REPLAN_EVERY, NAV_STUCK_STEPS, NAV_MAX_STUCK = 90, 120, 3
NAV_TIMEOUT, REACH_TIMEOUT = 1500, 400
STALL_STEPS, STALL_GAIN_M = 80, 0.005   # a reach whose error has not dropped by STALL_GAIN_M for STALL_STEPS controls is given up (0: off)
SCAN_RATE = 0.5                    # rad/s of the search turn
SCAN_MIN_TURN_RAD = 3.3            # the first turn is at least this long before planning (map for the routes)
SCAN_FULL_TURN_RAD = 6.6           # without both objects the search turns this far, then walks to a survey point
SCAN_WANT_TRIPOD = True            # the first turn goes on (up to SCAN_FULL_TURN_RAD) until the tripod is seen too
SURVEY_STEP_M = 1.2
MAX_SURVEYS = 4
SETTLE_STEPS = 12
LOOK_STEPS = 20                    # controls standing still at a stand to refresh an estimate
REST_TOL, REST_TIMEOUT = 0.05, 150   # the arms are at rest (largest joint error, rad) before a refit at a grasp stand
REFRESH_HINT, HINT_R_M = True, 0.5   # a refit after refresh() takes only candidates this near the forgotten estimate
KEEP_RATIO, KEEP_STEPS = 0.3, 30   # a new fit on fewer than KEEP_RATIO of the points of a fit made KEEP_STEPS ago does not replace it

# ---- arm ----------------------------------------------------------------------------------------------------------------------------------
REST_STEP = 0.02
COMPACT_ARMS = True                # travel and scan with the arms hanging (see phases/base.py rest_pose)
GRASP_RINGS = (0.45, 0.55, 0.65, 0.75, 0.85)
MOUNT_RINGS = (0.50, 0.58, 0.66, 0.74, 0.82, 0.90)
N_BEARINGS = 24
YAW_OFFSETS = (0., 0.35, -0.35, 0.7, -0.7)
MAX_STANDS = 400
STAND_FAR_COST = 3.0               # per metre the current pose lies outside the stand rings (a walk is cheaper than a stretched reach)
ROUTE_CHECKS = 25
SIDE_COST = {'left': 0.0, 'right': 0.3}
TILTS = (0.0, 0.35, 0.61)
APPROACH_SPEED, LIFT_SPEED, LINE_SPEED, INSERT_SPEED = 0.003, 0.003, 0.004, 0.001   # m per control
GRASP_SPREAD_M = 0.045             # each finger this far from the midpoint while approaching
LIFT_M = 0.15
BACKAWAY_M, BACKAWAY_TRIGGER_M, BACKAWAY_TIMEOUT = 0.35, 1.2, 200   # after the lift, back this far from a tripod nearer than the trigger (0: off)
GRIP_WAIT, HOLD_SETTLE = 60, 8
DESCEND_SOLVE = True               # descend by solve_near joint targets a little ahead on the line (not only the resolved-rate servo)
DESCEND_SOLVE_EVERY, DESCEND_LEAD_M = 10, 0.03
HELD_GAP_M = 0.012                 # closed fingers further apart than this hold something
CARRY_VMAX, CARRY_WMAX = 0.25, 0.3
PRE_M = 0.08                       # pre-mount: M this far above F
INSERT_BELOW_M = 0.012             # the insert line ends this far below F
INSERT_DWELL = 40
MOUNT_SOLVE = True                 # reach the pose over the mount by a solve_near joint target first, then the straight line
PRE_MOUNT_TIMEOUT = 300            # controls before the pose over the mount counts as unreachable for this attempt
DITHER = ((0., 0.), (0.012, 0.), (-0.012, 0.), (0., 0.012), (0., -0.012))
DITHER_YAW = (0.,)                 # yaw offsets (rad) of the mount attempts about F's vertical, the first is the estimate
MAX_GRASP_TRIES, MAX_MOUNT_ROUNDS = 10, 4
TUCK_X, TUCK_Y, TUCK_UP = 0.38, 0.22, 0.06
RETUCK = True                      # tuck again with the mount orientation once the tripod is found after the first tuck
LEG_CLEAR_M = 0.10                 # the fingers' approach line and the arm path keep this far from the tripod legs and column
ARM_PATH_CHECK = True              # the joint path to the pre-grasp point keeps the arm links LEG_CLEAR_M from the tripod
PATH_SAMPLES = 12
