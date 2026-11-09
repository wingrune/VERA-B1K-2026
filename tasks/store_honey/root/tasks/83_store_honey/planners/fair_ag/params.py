"""Constants of the fair planner of task 83 (mutated separately from the logic). Units: metres, radians, controls (30 Hz)."""

# ---- object priors (cabinet asset, see data/PROVENANCE.json) -------------------------------------------------------------------------------
CAB_LENGTH_M, CAB_DEPTH_M = 1.085, 0.476          # carcass top, along the front and across it
CAB_TOP_BAND_M = (0.79, 0.90)                       # height (robot frame, the floor is near 0) in which the cabinet top is searched
CAB_LENGTH_TOL, CAB_DEPTH_TOL = 0.20, 0.14          # a top cluster within this of the prior sizes is the cabinet
CAB_MIN_LENGTH_M = 0.55                             # a partly seen top this long along its long side is accepted as the cabinet (not complete)
CAB_MIN_FILL = 0.40                                 # share of the oriented rectangle covered by top cells (a top seen past a door edge is a triangle)
CAB_LOCK_M = 0.5                                    # once the whole top was seen, later clusters farther than this from it are other furniture
CAB_LENGTH_AGREE_M = 0.07                           # a top this close to the prior length is complete; a complete estimate moves only by such views
CAB_SMOOTH = 0.3                                    # share of a new complete view in the estimate
CAB_TRACK = False                                   # every complete view pulls the position, not only views of better length
CAB_LENGTH_GOOD_M = 0.03                            # views this close to the prior length are averaged into a complete estimate
ABOVE_GAP_M, ABOVE_MAX_M = 0.04, 1.6                # top-band cells with points this far above the band (walls, frames) are not a top
LEDGE_LATERAL_M = 0.4275                            # pull ledges either side of the drawer centre along the front
LEDGE_BELOW_TOP_M = 0.0545                          # ledge centre below the carcass top
LEDGE_OUT_OF_FACE_M = 0.015                         # pinch point out of the drawer face (the ledge sticks out 3 cm)
DRAWER_FACE_OUT_M = 0.017                           # drawer face out of the carcass top edge when the face is not measured
FACE_BAND_M = (0.25, 0.76)                          # heights of the drawer faces used to measure the front plane
FACE_WINDOW_M = (-0.06, 0.09)                       # window around the top edge (along f) where face points are looked for
FACE_MIN_POINTS = 60

JAR_DIAMETER_M, JAR_HEIGHT_M = 0.082, 0.151             # jar asset (data/PROVENANCE.json)
JAR_GRASP_ABOVE_M = 0.080                           # finger midpoint above the jar bottom at the grasp
HULL_OUT_OF_CENTRE_M = 0.016                        # top drawer cavity centre out of the carcass centre along f when closed (it moves with the drawer)
HULL_BELOW_TOP_M = 0.100                            # top drawer cavity centre below the carcass top; the cavity is 0.108 m high
HULL_FLOOR_BELOW_TOP_M = 0.154                      # top drawer floor below the carcass top

# ---- perception -------------------------------------------------------------------------------------------------------------------------
PERCEIVE_EVERY = 5
CLOUD_STRIDE = 4
CLOUD_RANGE_M = 4.5
SELF_RADIUS_M = 0.10
BODY_RADIUS_M = 0.42
MAP_RES_M, MAP_HALF_M = 0.025, 10.0                 # obstacle layers in the odometry frame
OBST_BAND_M = (0.05, 1.45)                          # heights that block the base (chair legs up to the counter overhang)
FLOOR_BAND_M = 0.035
OBST_HIT, OBST_FREE, OBST_MAX, OBST_MIN = 2, 1, 12, 3   # per frame: +HIT where an obstacle is seen, -FREE where only floor is seen; blocked at MIN
VISITED_RADIUS_M = 0.30
TOP_RES_M, TOP_DECAY, TOP_MIN = 0.02, 0.85, 0.6     # cabinet top layer: decays every perception, cell counts from TOP_MIN
CAB_HOLD_STEPS = 900                                 # an estimate not refreshed for this long is still used (the cabinet does not move)

HEIGHT_RES_M = 0.02                                 # height layer (top of the last points per cell) read by the arm clearance check
JAR_RES_M = 0.02
JAR_SUPPORT_MIN_M = 0.45                            # the jar stands on furniture, not on the floor
JAR_TOP_MAX_M = 1.60
JAR_RANGE_M = 4.0
PATCH_STEP_M = 0.03                                 # neighbouring cells of one surface differ by less than this in height
JAR_HEIGHT_TOL_M = 0.045
JAR_RING_CELLS, JAR_RING_KNOWN, JAR_RING_FLAT = 3, 0.35, 0.6
JAR_LOW_PERCENTILE = 25
JAR_VIEW_SHIFT_M = 0.015
JAR_MIN_CELLS, JAR_SPREAD_M = 5, (0.015, 0.029)
JAR_DENSE_STRIDE = 2                # head cloud stride of the jar search (the layers keep CLOUD_STRIDE)
JAR_GROUP_M, JAR_KEEP, JAR_MIN_SIGHTINGS, JAR_MAX_SIGHTINGS = 0.08, 12, 3, 400

# ---- heading from the walls ------------------------------------------------------------------------------------------------------------
HEADING_RANGE_M, HEADING_MIN_POINTS, HEADING_MIN_CONFIDENCE = 5.0, 600, 0.45
HEADING_GAIN, HEADING_MAX_STEP = 0.5, 0.05
LIFT_Z_MIN_M, LIFT_MAX_M = -0.8, 0.05

# ---- navigation ---------------------------------------------------------------------------------------------------------------------------
NAV_RES_M = 0.025                    # the doorway to the cabinet leaves a few centimetres per side to the base
BASE_FRONT_M, BASE_REAR_M, BASE_HALF_WIDTH_M, BASE_MARGIN_M = 0.25, 0.41, 0.36, 0.02
YAW_BINS, TURN_COST, HEURISTIC_WEIGHT = 16, 0.12, 1.4
NAV_REGION_M = 1.5
UNKNOWN_BLOCKS = False               # cells never seen as floor are free (the head sees no floor near the base)
NAV_GAIN, NAV_VFAR, NAV_VNEAR, NAV_YAW_GAIN, NAV_WMAX, NAV_WMIN = 0.9, 0.35, 0.15, 1.0, 0.5, 0.12
NAV_POS_TOL, NAV_YAW_TOL = 0.04, 0.06
LOOKAHEAD_M = 0.30
NAV_SHRINKS = (0.0, 0.03)      # footprint narrowing (m per side) tried in turn when the full footprint has no route
CLEAR_M, CLEAR_COST = 0.12, 3.0      # clearance penalty: poses closer than CLEAR_M to a blocked pose cost up to CLEAR_COST more per move
START_ESCAPE_M = 0.10                # a start pose inside the grown obstacles may move this far through them
CARRY_VFAR, CARRY_VNEAR, CARRY_WMAX = 0.10, 0.08, 0.25
REPLAN_EVERY, STUCK_STEPS, NO_ROUTE_MAX = 150, 120, 40
SCAN_RATE = 0.6                      # rad/s of the search turn
SCAN_MIN_TURN_RAD = 6.0
SETTLE_STEPS = 12
EXPLORE_STEP_M, EXPLORE_TRIES = 1.2, 4

# ---- drawer ---------------------------------------------------------------------------------------------------------------------------
DRAWER_STANDS = (0.55, 0.50, 0.62)   # base distance in front of the ledge along the pull direction
DRAWER_LATERALS = (0.30, 0.25, 0.35, 0.20)   # base offset from the ledge towards the cabinet centre (the hand on the ledge side)
PRE_GAP_M = 0.12
PINCH_FINGER_BAND = (0.002, 0.011)   # finger position (m) of a pinch closed on the 1.3 cm ledge
GRIP_WAIT = 45
PULL_SPEED = 0.15                    # fraction of the full base speed
DRAWER_OPEN_TARGET_M = 0.36          # the hand moves with the base, so the base travel is the drawer travel
PULL_STALL_STEPS = 60
PULL_MAX_M = 0.55
RETREAT_HAND_M, RETREAT_BASE_M = 0.10, 0.22
REST_STEP = 0.02
MAX_DRAWER_TRIES = 6

# ---- jar ------------------------------------------------------------------------------------------------------------------------------
FIND_JAR_TURN_RAD = 6.6              # in-place search turn for the jar when it was not seen yet
JAR_RINGS = (0.65, 0.75, 0.85, 0.95, 1.05)
JAR_BEARING_STEP_RAD = 0.1745
JAR_PRE_GAP_M, JAR_DEEP_M = 0.12, 0.02
JAR_LOOK_STEPS = 40                  # controls standing at the jar stand while the head re-measures the jar
JAR_LIFT_M = 0.16
JAR_FINGER_BAND = (0.030, 0.046)     # finger position closed on the 8 cm jar
JAR_HELD_MIN = 0.018                 # fingers closer than this hold nothing
CARRY_POINT = (0.30, 0.12)           # carry posture of the finger midpoint, robot frame (forward, to the side of the hand)
CARRY_ROLL = 1.2                     # rad about the finger axis after the side pinch closed: the lower finger moves under the jar
ARM_CLEAR_M = 0.03                   # arm links stay this far above the perceived top under them
CAND_LIMIT = 12
SCAN_STANDS = 400

# ---- place ----------------------------------------------------------------------------------------------------------------------------
PLACE_STANDS = (0.55, 0.62, 0.70, 0.78)
PLACE_LATERALS = (0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3)
PLACE_ABOVE_M = 0.17
PLACE_CLEAR_M = 0.012                # jar bottom above the drawer floor at the place target
LOCK_TOL = 0.02
VIEW_DISTANCES_M = (1.2, 1.0, 1.4)    # a stop this far in front of the cabinet centre before the drawer stand: the estimate from the scan is off
ODO_CORRECTED = True                  # base translation integrated along the wall-corrected heading
VIEW_STEPS = 40                       # controls of looking at the cabinet there (the head sees its whole top)
TIMEOUT = dict(settle=SETTLE_STEPS + 5, scan=900, explore=900, go_view=1500, drawer_view=120, plan_drawer=10, go_drawer=2400, drawer_pre=500, drawer_reach=400, drawer_grip=120,
               drawer_pull=500, drawer_release=60, drawer_retreat=500, find_jar=900, plan_jar=10, go_jar=2400, jar_look=200, jar_pre=600,
               jar_reach=500, jar_grasp=120, jar_lift=300, jar_retract=1500, plan_place=10, go_place=2400, place_above=600, place_lower=500,
               place_release=200, place_retreat=300)
MAX_JAR_TRIES, MAX_PLACE_TRIES = 6, 6
