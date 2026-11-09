"""Constants of the fair planner of task 42 (mutated separately from the logic). Units: metres, radians, controls (30 Hz).

Heights are measured from the floor seen by the head camera (perception estimates the floor height in the robot frame)."""

# ---- perception -------------------------------------------------------------------------------------------------------------------------
PERCEIVE_EVERY = 5                 # controls between perception updates
HEAD_STRIDE, WRIST_STRIDE = 2, 3   # pixel strides of the back-projection
CLOUD_RANGE_M = 5.0                # points farther than this (horizontal) from the base are dropped
SELF_RADIUS_M = 0.085
FINGER_RADIUS_M = 0.05
BODY_RADIUS_M = 0.36
TORSO_CUT = True
TORSO_RADIUS_M = 0.17
RECENT_UPDATES = 8                 # perception updates kept in the recent cloud (detection)
FLOOR_BAND_M = 0.02
OBST_RES_M, OBST_HALF_M = 0.05, 12.0
OBST_BAND_M = (0.06, 1.7)
OBST_MIN_HITS = 2
OBST_SELF_R_M = 0.5
UNKNOWN_COST = 1.0

# counter and basin
COUNTER_BAND_M = (0.85, 0.91)      # the counter top (static map prior: 0.88 m) lies in this height band above the floor
COUNTER_FLAT_M = 0.012             # points this near the peak height belong to the slab
COUNTER_MIN_POINTS = 600
COUNTER_RES_M = 0.015
COUNTER_MIN_AREA_M2 = 0.35
COUNTER_MIN_DEPTH_M, COUNTER_MAX_DEPTH_M, COUNTER_MIN_LENGTH_M = 0.45, 0.85, 0.8
COUNTER_OPEN_CELLS = 9             # structuring element (cells) that opens thin links off the counter slab
BASIN_RES_M = 0.01
BASIN_BELOW_M = (0.09, 0.15)       # the basin floor lies this far below the counter top (static map prior: 0.114 m); other recesses (hob grates) are shallower
BASIN_TOP_IN, BASIN_FLOOR_OUT, BASIN_TOP_RIM = 3.0, 3.0, 0.6   # template scores
BASIN_MIN_POINTS, BASIN_MIN_SCORE = 200, 150.0
BASIN_SIZE_M = (0.33, 0.44)       # prior (static map): basin extents along the counter and across it
ITEM_RES_M = 0.01
ITEM_BAND_M = (0.008, 0.14)        # items on the counter reach this far above its top
SINK_MARGIN_M = 0.02

# stall check: the jammed base still reports its commanded velocity; a pose move without a change of the head depth is undone
STALL_EVERY, STALL_STRIDE = 10, 24
STALL_MIN_MOVE_M, STALL_MIN_TURN_RAD = 0.04, 0.08
STALL_DEPTH_M, STALL_MAX_DEPTH_M = 0.01, 4.0

# ---- navigation -------------------------------------------------------------------------------------------------------------------------
ROBOT_RADIUS = 0.35
TIGHT_RADIUS = 0.30
GRID_RES = 0.05
NAV_GAIN, NAV_VFAR, NAV_VNEAR, NAV_YAW_GAIN, NAV_WMAX = 0.9, 0.35, 0.14, 0.8, 0.4
NAV_POS_TOL, NAV_YAW_TOL = 0.07, 0.18
NAV_REPLAN_EVERY, NAV_STUCK_STEPS = 90, 160
SETTLE_STEPS = 6
SCAN_RATE = 0.5

# items
FLAT_MAX_M = 0.04                  # board and parer are thinner than this above the counter top
PARER_SHORT_MAX_M, PARER_LONG_M = 0.075, (0.10, 0.34)
BOARD_SHORT_MIN_M, BOARD_LONG_M = 0.14, (0.22, 0.42)
BOWL_H_M, BOWL_SHORT_MIN_M, BOWL_LONG_MAX_M = (0.045, 0.12), 0.18, 0.36
ITEM_MIN_CELLS = 25
ONION_MIN_ABOVE_FLOOR_M, ONION_MIN_POINTS = 0.02, 40

# model fits (batched yaw grid + ICP on the GPU)
FIT_BACKEND = 'gpu'
FIT_YAW_STEP_DEG = 10.0
FIT_ITERS = 25
FIT_TRIM_M = 0.03
FIT_INLIER_M = 0.012
FIT_MAX_CLOUD = 500
FIT_MIN_POINTS = 25
FIT_MIN_INLIER = 0.5
FIT_ROI_PAD_M = 0.04
FIT_Z_BAND_M = (0.002, 0.04)       # fit points: this far above the counter top
SLOT_BOX_M = ((0.100, 0.114), (-0.035, 0.035))   # inside the board's handle slot (model frame: the slot is x 0.095-0.12, |y| < 0.045)
SLOT_MIN_H_M = 0.012       # hypotheses this near the best cost compete; a point above this height over the counter is board
SLOT_HIT_MARGIN = 4                               # the flipped board pose wins when its slot holds this many points fewer

# bowl contents
BOWL_INNER_R_M = 0.10              # the bowl interior seen from above (inner points reach 0.1 m from its axis)
BOWL_FLOOR_M = 0.027               # inner floor of the bowl above the counter top
PIECE_MIN_ABOVE_M, PIECE_MIN_POINTS = 0.012, 30

# ---- phases ---------------------------------------------------------------------------------------------------------------------------
REST_STEP = 0.02
LOOK_STEPS = 24                    # controls standing still with the arms at rest to refresh the estimates
VANTAGE_M = 1.25                   # base distance from the counter's front edge when it looks over the counter
SHIFT_M = 0.6                      # lateral step along the counter when something is missing in the view
MAX_SHIFTS = 4
SCAN_FULL_TURN_RAD = 6.6
MAX_SURVEYS = 8
FRONTIER_DILATE, FRONTIER_INFO_R_M, FRONTIER_INFO_GAIN = 4, 1.5, 3.0
FRONTIER_MIN_M, FRONTIER_MAX_M, FRONTIER_VISITED_M = 1.0, 8.0, 1.5
RINGS = (0.55, 0.62, 0.70, 0.78, 0.87, 0.95, 1.05, 1.15, 1.25, 1.35)
N_BEARINGS = 24
KEEP_YAW_RAD = 0.35
FRONT_MARGIN_M = 0.15              # stands lie at least this far in front of the counter's front edge (inflated footprint aside)
NAV_TIMEOUT = 1800
REACH_POS, REACH_ORI = 0.04, 0.55
CLOSE_DWELL, OPEN_DWELL = 18, 12
LIFT_RISE_M, HOLD_RISE_M = 0.24, 0.08
HELD_GAP_M = 0.012                 # closed fingers further apart than this hold something
MAX_TRIES = 4
PHASE_TIMEOUT = dict(reach=420, grasp=220, lift=360, carry=1600, lower=350, release=40, chop=160, clear=80, wait_slicer=120, dice=260,
                     sink_go=1400, sink_lower=420, chop_go=1800)
BOWL_NEAR_RISE_M = 0.07            # the finger midpoint over the bowl when the onion is let go: this far above the bowl's top
SLICER_WAIT = 75
CHOP_STEPS = 125
DICE_STEPS = 190
DICE_SWEEP_M, DICE_SWEEP_PERIOD = 0.035, 80
WHOLE_TOP_M = 0.08                 # a blob in the bowl higher than this above the counter top is the whole onion (halves reach about 0.065)
SINK_SPOTS = ((0., 0.), (0.05, 0.), (-0.05, 0.), (0., 0.05), (0., -0.05))    # release spots over the basin: offsets (along, across) from its centre
SINK_Z_BELOW_TOP_M = 0.02
BASIN_DWELL = 40
MOVING_SPEED = 0.03                # base speed (m/s or rad/s of the proprio) above which the base counts as moving
BAD_STAND_M = 0.25                # a stand within this of a stand where a pick failed is not offered again
BASIN_ITEM_MIN_M, BASIN_ITEM_BELOW_TOP_M = 0.012, 0.03   # something in the basin: above its floor by this and below the counter top by this
ONION_WALL_M = 0.03                # the basin walls are not the onion: stay this far inside the basin box

# anchoring the odometry on the counter (heading, front edge, basin)
ANCHOR_MIN_LENGTH_M = 1.2          # a view of the counter at least this long anchors
ANCHOR_GAIN = 0.0                  # anchoring switched off: it diverged between runs; the stands follow the counter seen now instead (phases/pick.py go)
ANCHOR_YAW_DEAD_RAD, ANCHOR_MAX_YAW_RAD = 0.008, 0.6
ANCHOR_POS_DEAD_M, ANCHOR_MAX_POS_M = 0.010, 1.0
ONION_R_M, ONION_CAP_M = 0.0345, 0.025   # onion radius (asset prior); the cap whose mean xy is the onion's xy
STALE_OK_STEPS = 2500              # an estimate this young is used when the object is hidden from the stand (the odometry is anchored on the counter)
LOOK_SETTLE_STEPS, LOOK_EXTRA_STEPS = 12, 60   # a look ends this long after the last anchoring correction (at most this many extra controls)
REFINE_R_M, REFINE_MIN_POINTS = 0.10, 60       # the wrist view of the onion from the pregrasp: radius around the estimate, least points
REFINE_STEPS, REFINE_MAX_SHIFT_M = 12, 0.10
CLEAR_SLACK_M = 0.035               # a sphere may sink this far into the counter slab before it counts as a hit (the spheres are coarse: only gross hits reject a stand)
GRASP_ARRIVE_M, GRASP_MAX_AGE = 0.02, 200   # the fingers close when the finger midpoint is this near the grasp point (or after this many controls)
OBST_DECAY, OBST_MAX_HITS = 0.97, 12.0       # per perception update (5 controls); a cell must be seen again within about 20 s
ANCHOR_CLEAR_YAW_RAD, ANCHOR_CLEAR_POS_M = 0.02, 0.05   # a correction this large clears the obstacle hits registered before it

# stand layout in the counter frame
STAND_DISTS = (0.36, 0.42, 0.50, 0.60, 0.72, 0.86)   # base distance from the counter's front edge
STAND_SPAN_M, STAND_STEP_M = 0.60, 0.05              # along the counter, either side of the target
STAND_YAWS = (0.0, 0.3, -0.3)                        # yaw offsets from facing the counter
STAND_MAX_SCAN = 400
COUNTER_CLEAR_FRONT_M = 0.12                         # obstacle hits this far in front of the edge belong to the counter front
FINGER_OFFSET_M, JAW_TRIES = 0.058, 4   # open fingers: centre offset from the grasp point along the jaw, half thickness; jaw yaws tried
ANCHOR_TOTAL_YAW_RAD = 1.5          # the anchoring never turns the pose by more than this in total (a runaway guard)
FOLLOW_FRESH_STEPS = 60              # the counter estimate counts as seen now when it is this young
COUNTER_LOCK_RAD, COUNTER_LOCK_STEPS = 0.6, 900   # a slab turned by more than this against the counter seen in the last COUNTER_LOCK_STEPS controls is another run of counters
CLEAR_BASIN_SHRINK_M = 0.0         # a sphere centre inside the basin box passes (the basin estimate is a few cm off: the sphere radius is not taken off the box)
LAYERS_MAX_TURN_RATE = 0.06         # rad/s: faster turns do not write obstacle hits
SCAN_TURN_CTRL, SCAN_PAUSE_CTRL = 60, 14   # scan: turn this many controls (0.5 rad/s: 1 rad), then stand still this many
SCAN_PARTIAL_TURN_RAD = 4.0          # at a frontier the scan turns this far (the head sees 100 degrees at a time)
BASIN_PTS_R_M = 0.40
FINGER_BLOCK_ABOVE_M = 0.025          # a cloud point this far above the basin floor is something a finger would hit
FINGER_HALVES_M = (0.012, 0.02, 0.03)   # footprint half sides tried for the finger room
FLOOR_Z_M = -0.005                  # the floor in the robot frame (base_link origin height above the floor)
WALK_MAX_STEP = 0.024               # rad per control of the joint walk to the scan's solution
REACH_SWITCH_AGE, REACH_SWITCH_ERR = 120, 0.15   # reach: solve_near that has not got within this of the pregrasp by this age is replaced by the joint walk
REACH_WAYPOINT_STEPS = 260   # reach: walk mode lasts this many steps, then the servo (constant lost in ef6bd61: walk mode crashed with AttributeError on 301, 306)
STAND_MAX_DIST_M = 0.95             # stands farther than this (horizontally) from the target are not offered
START_FREE_M = 0.30                # the cells within this of the base are free in the route grid (the robot is there)
