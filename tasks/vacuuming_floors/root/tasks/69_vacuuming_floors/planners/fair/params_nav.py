"""Navigation and motion constants of the fair planner of task 69: routes, the search turn, joint walk speeds, base speeds (re-exported by params.py)."""

ROBOT_RADIUS = 0.34
RELAX_RADIUS = 0.28                # every route failed: retry once with this footprint radius (0: no retry)
GRID_RES = 0.05
NAV_GAIN, NAV_VFAR, NAV_VNEAR, NAV_YAW_GAIN, NAV_WMAX = 0.9, 0.4, 0.15, 0.8, 0.5
NAV_POS_TOL, NAV_YAW_TOL = 0.03, 0.04
NEAR_POS_TOL, NEAR_YAW_TOL, NEAR_STALL_STEPS = 0.06, 0.10, 45
NAV_REPLAN_EVERY, NAV_STUCK_STEPS, NAV_MAX_STUCK = 90, 120, 3
NAV_TIMEOUT = 1200
SCAN_RATE = 0.9                    # rad/s of the search turn
SCAN_TURN_RAD = 6.4                # the first turn is this long (the head sees the floor ring 0.9-3 m around the robot)
SURVEY_STEP_M = 1.2
MAX_SURVEYS = 4
SETTLE_STEPS = 12
REST_TOL, REST_TIMEOUT = 0.05, 150   # the arms are at rest (largest joint error, rad) before a refit at a grasp stand
REST_STEP = 0.02
JOINT_STEP = 0.012                 # rad per control towards a joint target (all joints arrive together)
WALK_SETTLE = 8
LINE_STEP_M = 0.01
SWEEP_VFAR, SWEEP_VNEAR, SWEEP_WMAX = 0.40, 0.15, 0.40
SWEEP_POS_TOL, SWEEP_YAW_TOL = 0.02, 0.06
SWEEP_TIMEOUT = 600
ROBOT_RADIUS_TIGHT = 0.28
HEAD_HALF = (0.17, 0.15)           # head footprint half extents around the cube centre (along the cube's long axis, across)
HEAD_MARGIN_M = 0.02
SCAN_STOP_RAD = 0.9                # the search turn goes in steps of this angle, standing still between them
SCAN_PAUSE = 12                    # controls standing still after each step of the turn
PRE_STEP = 0.03                    # rad per control of the long joint walks (to the pre-grasp point, to the press, home)
APPROACH_STEP = 0.008              # rad per control of the short moves at the handle
WAYPOINT_M = 0.35                 # the route to a grasp stand ends this far out on the line from the vacuum through the stand, the last leg is straight in
ACCEL_LIN, ACCEL_ANG = 0.03, 0.05       # largest change per control of the base command (native units, 1 = 0.75 m/s or 1 rad/s) while the vacuum is held
SWEEP_STALL_STEPS, SWEEP_PROGRESS_M = 150, 0.03      # a drive that gets no closer than this in that many controls is given up (another pass direction next)
SWEEP_GAIN = 1.8                   # proportional gain of the drive to a flake (the approach is exponential: at 0.9 the last 0.3 m took about 120 controls)
