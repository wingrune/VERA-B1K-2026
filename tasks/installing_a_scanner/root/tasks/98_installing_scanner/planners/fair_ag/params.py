"""Constants of the fair planner of task 98 (mutated separately from the logic). Units: metres, radians, controls (30 Hz)."""

# ---- scene priors (object templates of the task, see data/PROVENANCE.json) -------------------------------------------------------------------
SCANNER_LONG_M, SCANNER_SHORT_M = 0.311, 0.218     # collision box of the scanner asset, long side carries the button
SCANNER_MARKER_FRONT_M = 0.1485                    # button centre in front of the box centre along the long axis
SCANNER_MARKER_BELOW_TOP_M = 0.0245                # button centre below the top face of the scanner
SCANNER_HEIGHT_MIN_M = 0.050                       # a desk object this much above the desk top is the scanner, lower flat ones the laptop
LAPTOP_HEIGHT_RANGE_M = (0.015, 0.048)
LAPTOP_EXTENT_M = (0.22, 0.50)                     # a flat desk object this wide on both sides is the laptop
LAPTOP_AGREE_M = 0.12                              # later laptop clusters must lie this close to the stored box centre
TOP_MIN_VOTES, TOP_RESET_M = 600, 0.015           # votes before a desk top height is accepted; a shift larger than this clears the surface layer
DESK_TOP_RANGE_M = (0.55, 1.0)                     # height (robot frame) in which the desk top is searched

# ---- perception -------------------------------------------------------------------------------------------------------------------------
PERCEIVE_EVERY = 5               # controls between full perception updates
CLOUD_STRIDE = 4
CLOUD_RANGE_M = 4.0              # points farther than this from the robot are not used
SELF_RADIUS_M = 0.10             # points this near an arm link are the robot's own
BODY_RADIUS_M = 0.42             # points this near the base axis (horizontal) are the robot's own
OBST_RES_M, OBST_HALF_M = 0.05, 12.0
OBST_BAND_M = (0.05, 1.35)        # includes the legs and castors of chairs (the base climbs them: base height rose by 0.2 m in an early run)
OBST_MIN_HITS = 3
FLOOR_BAND_M = 0.04              # head points below this height are floor: that cell is known free ground
FLOOR_CLOSING = 2
VISITED_RADIUS_M = 0.45
BLIND_ZONE_M = 1.0               # the head camera sees no floor nearer than this: cells this close to the base count as free
SOFT_CLUTTER_MAX_M = 2.2             # chairs side by side form one low cluster; walls and partitions reach above the obstacle band, the desk is cut out
DESK_CUT_MARGIN_M = 0.06
UNKNOWN_COST = 1.5               # extra path cost factor of cells never seen as floor
SOFT_DEPTH_COST = 8.0            # extra cost per metre of depth inside a chair: a chair is crossed along its edge, not through its middle
SOFT_DESK_COST = 4.0             # extra cost per metre between a chair cell and the desk
SOFT_COST = 0.3                  # extra path cost factor of a chair the base would have to push
TOP_RES_M = 0.02
TOP_BAND_M = 0.012
DESK_MIN_AREA_M2 = 0.8           # desk top seen at least this much before the frame is aligned to it
DESK_LOCK_AREA_M2 = 1.5
RED_MIN_R, RED_MAX_GB_FRAC, RED_MIN_PIXELS = 80, 0.25, 4
BLOB_FRESH_STEPS, BLOB_AGREE_M = 40, 0.04        # a red blob younger than this and nearer the scanner template than this replaces the template as the button
OBJ_RES_M = 0.01
OBJ_MIN_CELLS = 12
SCANNER_JUMP_M = 0.5
SCANNER_TRACK_STEPS = 60          # while the scanner was seen this recently, a new measurement may change its pose only by the steps below
SCANNER_YAW_STEP, SCANNER_SHIFT_STEP_M = 0.03, 0.03
SCANNER_MIN_COVER, SCANNER_MIN_POINTS = 0.3, 25   # a partly hidden scanner still moves the estimate when its visible part covers this much

# ---- position from the wall planes -----------------------------------------------------------------------------------------------------
WALL_BIN_M, WALL_AXIS_TOL_DEG, WALL_MIN_VOTES, WALL_MIN_VIEW_POINTS = 0.02, 12.0, 400, 150   # offset histogram bin, normal tolerance, votes of a recorded plane, points of a plane in a view
WALL_MATCH_M, WALL_GAIN, WALL_MAX_STEP_M = 0.20, 0.1, 0.005                                  # observed plane matches a recorded one within this; share and step of the correction
WALL_RECORD_STEPS = 270                                                                      # the planes are recorded during the first scan turn (controls)

# ---- heading from the walls ------------------------------------------------------------------------------------------------------------
HEADING_RANGE_M, HEADING_MIN_POINTS, HEADING_MIN_CONFIDENCE = 5.0, 600, 0.45   # wall points used, how many of them, how peaked the direction vote must be
HEADING_GAIN, HEADING_MAX_STEP = 0.6, 0.3                                      # share of the heading error corrected per update, and the largest step (rad)

LIFT_Z_MIN_M, LIFT_MAX_M = -0.8, 0.05   # floor heights searched for the lift estimate; a lift above LIFT_MAX_M means the base climbed onto something
BUMP_RADIUS_M = 0.2

# ---- navigation -------------------------------------------------------------------------------------------------------------------------
ROBOT_RADIUS = 0.34
ROBOT_TIGHT_RADIUS = 0.28         # the true footprint radius of the base
START_RELAX_M = 0.9              # within this distance of the base the tight radius is used when planning from the current pose
GRID_RES = 0.05
NAV_GAIN, NAV_VFAR, NAV_VNEAR, NAV_YAW_GAIN, NAV_WMAX = 0.9, 0.5, 0.18, 0.8, 0.5
NAV_POS_TOL, NAV_YAW_TOL = 0.05, 0.08
NAV_VSOFT = 0.30                 # speed (m/s) while the route ahead crosses a chair
NAV_REPLAN_EVERY, NAV_STUCK_STEPS, NAV_MAX_STUCK = 90, 120, 3
SCAN_RATE = 0.7                  # rad/s of the search turn
SCAN_MIN_TURN_RAD = 5.8          # the first scan turns this far (a map of the room for localisation and for the route) before planning
SCAN_MAX_TURNS = 1.2             # full turns before the search gives up and walks
SETTLE_STEPS = 12

# ---- arm ----------------------------------------------------------------------------------------------------------------------------------
REST_STEP = 0.02
REACH_POS_TOL, REACH_ORI_TOL = 0.025, 0.30
REACH_TIMEOUT, PRESS_TIMEOUT, PUSH_TIMEOUT, NAV_TIMEOUT = 400, 700, 2400, 1500
PRESS_SPEED_M, PRESS_DWELL, DITHER_M = 0.001, 40, 0.010
PRESS_DEPTH_M = -0.02
REPRESS_ALLOWED = False
PRE_OFFSET_M = 0.12
TILTS = (0.6, 1.1)
RINGS = (0.55, 0.65, 0.75, 0.85)
BEARING_STEP_DEG = 20
CAND_LIMIT = 10

# ---- push -----------------------------------------------------------------------------------------------------------------------------------
PUSH_SPEED_M, PUSH_LEAD_M, PUSH_MAX_ROUNDS, PUSH_STALL_STEPS = 0.0012, 0.03, 10, 150
PUSH_PROGRESS_M = 0.01
PUSH_CAP_STEP_M, PUSH_CAP_EVERY, PUSH_CAP_STEPS = 0.01, 40, 4   # while the scanner does not move, the tip goal goes 1 cm deeper every 40 controls (up to 4 times)
PUSH_YAW_MAX = 0.3
PUSH_ACROSS_RATE_M = 0.0004
PUSH_EDGE_STOP_M = 0.02
PUSH_HEIGHT_M = 0.05              # pusher fingertip height above the desk top (the collision top of the desk lies 1.5 cm above the visible one)
PUSH_TILTS = (1.2, 1.45)
PUSH_RINGS = (0.55, 0.7, 0.85, 1.0, 1.15)
MIN_PUSH_M = 0.01
ARM_EXTRA_M = 0.25
GAP_GOAL_M = 0.04
EDGE_MARGIN_M = 0.04
