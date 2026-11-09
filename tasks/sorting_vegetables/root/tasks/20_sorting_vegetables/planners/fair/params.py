"""Constants of the fair planner of task 20 (mutable as one module). Lengths in metres, angles in radians, times in controls (30 Hz).
Heights are above the floor (the robot frame's z: the base stands on the floor)."""

# ---- perception: clouds ---------------------------------------------------------------------------------------------------------------
PERCEIVE_EVERY = 3            # controls between two perception updates (every control in the look phases)
HEAD_STRIDE, WRIST_STRIDE = 3, 4
BASE_RADIUS_M, BASE_TOP_M = 0.36, 0.45   # the base as a low cylinder (cut out of the clouds)
TORSO_RADIUS_M, SELF_RADIUS_M, FINGER_RADIUS_M = 0.17, 0.08, 0.035
MOVING_SPEED = 0.05           # base speed (m/s or rad/s) above which no detection is made (the views do not line up)
TURN_RATE_MAX_MAP = 0.35      # rad/s: no map update while turning faster (the heading of a turn is integrated a few per cent off)
ODO_TURN_SCALE = 1.0          # the measured turn rate is about right in continuous turns (survey drive of 302: 3.08 integrated, 3.04
# turned) and 4-7 per cent short in the stop-and-go scan (start and stop transients): the heading fixes take care of both
RELOC_MIN_SIGHTINGS, RELOC_MAX_M, RELOC_AGREE_M, RELOC_MIN_M, RELOC_GAIN = 10, 0.5, 0.06, 0.03, 0.5   # base position from >= 2 bowls
STILL_STRIDE, STILL_DEPTH_M, STILL_VEL = 24, 0.002, 0.03   # head depth grid stride, mean change under which the base stands still
LOC_MAX_TURN = 0.15           # rad/s: heading fixes while the base turns slower than this (it may translate)
LOC_EVERY = 12                # controls between two heading fixes while the base stands
LOC_BAND_M = (0.30, 1.40)     # heights of the points that give the house's axes (furniture and walls, not the floor or the vegetables)
LOC_RANGE_M = 4.0
LOC_MIN_POINTS, LOC_MAX_POINTS = 200, 4000
AXES_STEP_DEG, AXES_BIN_M = 0.25, 0.02
AXES_MIN_PEAK = 2.0           # the sharpest projection this many times the median one (3.2-9 on every standing view of the scans)
AXES_FIX_STEPS = 40           # views up to this control fix the axes in the odometry frame (the base stands while the trunk settles)
AXES_MAX_CORR = 0.15          # rad: a larger correction needs two views that agree within AXES_AGREE_RAD (up to 45 degrees)
AXES_AGREE_RAD = 0.04
# ---- occupancy ------------------------------------------------------------------------------------------------------------------------
MAP_HALF_M = 9.0              # the map spans this much round the start in x and y
VOX_M, VOX_TOP_M = 0.05, 2.0
VOX_DECAY, VOX_MAX_HITS, VOX_MIN_HITS = 0.997, 12.0, 2.0
FLOOR_BAND_M = 0.04           # points within this of the floor are floor
NAV_BAND_M = (0.08, 1.30)     # an occupied voxel in this band blocks the base (the arms and the torso stick out up to the counter top)
ROBOT_RADIUS = 0.30           # the obstacles grown by this for routes (oracle: 0.34 closed a 0.65 m gap the base fits through; with 0.33 the
# perceived counter closed the way to the stands by bowl_1 of 302, t20f7)
ROBOT_RADIUS_WIDE = 0.40
VIEW_RANGE_M = 4.0
FOOTPRINT_CLEAR_M = 0.25      # the occupancy under the standing base is cleared (it cannot stand in an obstacle)
BODY_MARGIN_CELLS = 1         # body points this many voxels from an occupied one hit it            # map and detections only from head points closer than this

# ---- perception: colours (sRGB 0-255 of the native renders; data/PROVENANCE.json says where they were measured) ---------------------
RED_MIN, RED_G_MAX, RED_B_MAX = 45, 0.48, 0.50     # bowls (and the backsplash, the knobs): R >= RED_MIN, G <= RED_G_MAX * R, B <= RED_B_MAX * R;
# the H100 port renders the bowls 2-4x darker (R median 70 against 194 native): R >= 90 kept 15 % of their pixels there (61-63 % now on both)

# ---- bowls ----------------------------------------------------------------------------------------------------------------------------
COUNTER_TOP_M = 0.885         # prior: the countertop height above the floor (one kitchen); refined from the cloud
BOWL_BAND_M = (0.86, 1.09)    # red points of a bowl on the counter lie in this height band
BACKSPLASH_ABOVE_M = 1.12     # a 4 cm column with red above this is the backsplash, not a bowl
BOWL_RIM_R_M = 0.155          # rim radius of the mixing bowl (asset)
BOWL_RIM_BAND_M = (1.030, 1.062)   # the top of a red blob must be here (rims 1.043-1.052 at 1-2.4 m; something red by bowl_3 of 302 at 1.067)
BOWL_MERGE_EXT_M = 0.40      # blobs whose union spans at most this in x and y are one bowl
BOWL_EXT_M = (0.10, 0.42)     # extent of a red blob that can be a bowl
BOWL_MIN_POINTS = 40
BOWL_MATCH_M = 0.25           # a sighting within this of a bowl track updates it (bowls are 0.45 m apart or more)
BOWL_RELOOK_M = 0.15          # from the drop stand, the bowl seen within this of the planned one is the target (a blob of one bowl's
# far wall and the next one's near rim made a bowl 0.25 m off both, native t20f9)

# ---- motion ---------------------------------------------------------------------------------------------------------------------------
SETTLE_STEPS, SETTLE_MAX = 12, 150   # the trunk walks from its start pose to the rest pose (the head looks down)
REST_STEP = 0.02
TUCK_ARM = (0.15, 0., 0., 0.09, 0., 0., 0.)   # the idle arm along the torso (oracle finding: at rest the arms sweep bowls and stools)
SCAN_RATE = 0.45              # rad/s of the turn in place
SCAN_TURN_RAD = 6.6           # a full turn and a bit
NAV_GAIN, NAV_VFAR, NAV_VNEAR, NAV_YAW_GAIN, NAV_WMAX = 0.9, 0.45, 0.15, 0.8, 0.5
NAV_LOOKAHEAD_M = 0.3
NAV_POS_TOL, NAV_YAW_TOL = 0.08, 0.12
NEAR_POS_TOL, NEAR_YAW_TOL, NEAR_STALL_STEPS = 0.20, 0.60, 40
NAV_REPLAN_EVERY, NAV_STUCK_STEPS = 60, 120
NAV_TIMEOUT = 1500
NAV_MPS_STEP, NAV_SLACK = 0.007, 300
ESCAPE_STEPS, ESCAPE_SPEED = 40, 0.15
SCAN_TURN_CTRL, SCAN_PAUSE_CTRL = 20, 12   # turn this many controls, then stand this many (detections are made standing)
MAX_SCANS = 6
RETRY_WAIT = 60               # controls in done/failed before the search starts over
SURVEY_MIN_M, SURVEY_MAX_M = 1.0, 3.0
BOWL_MIN_SIGHTINGS = 2        # a bowl track counts once seen this often
BOWL_APART_M = 0.35           # distinct bowls are at least this far apart (0.45 m between two bowls of 302)
BASKET_MIN_SIGHTINGS = 2
VIEW_RINGS = (0.95, 1.1, 1.25)   # stands to look into a basket from
VIEW_MAX_ROUTE_M = 5.0

# ---- vegetables and baskets (colour thresholds measured on native frames, data/PROVENANCE.json) -----------------------------------------
CORN_R_MIN, CORN_G_FRAC, CORN_B_FRAC = 150, (0.45, 0.95), 0.30   # corn 78%, bok choy 4%, wicker 5% (husk pixels outside the corn hull), floor 0.4% (t20f2 302)
ONION_R_MIN, ONION_G_MAX, ONION_GB_MAX = 120, 0.80, 0.08   # pastel pink: G and B both about 0.67 R, close to each other (wicker: B 0.2 R
# under G); onions 71%, wicker 3.5%, floor 0.08% (t20f5 302, basket 2 views)
GREEN_G_MIN, GREEN_OVER_R, GREEN_OVER_B = 50, 0.97, 1.15   # leeks 71%, broccoli 62%, bok choy 69% (white stems are not green), floor 0.2%
WICKER_R_MIN, WICKER_G_FRAC, WICKER_B_FRAC = 70, (0.70, 0.92), (0.45, 0.80)
VEG_Z_M = (0.015, 0.45)       # vegetables on the floor or in a basket
VEG_RES_M, VEG_MIN_POINTS = 0.015, 12
VEG_MAX_LEN_M = 0.45
VEG_MIN_WIDTH_M = 0.025       # narrower blobs are edges (vegetables 0.04-0.12 m wide)
VEG_MATCH_M = 0.08
VEG_EXPIRE_TAN, VEG_EXPIRE_RANGE_M, VEG_MISSES = 0.85, 2.0, 4   # well inside the head view (99 degrees: tan 1.17), close, views
VEG_EXPIRE_EVERY = 15         # controls between two expiry checks (look phases detect every control)
BASKET_TOP_M, BASKET_RES_M, BASKET_MIN_POINTS = 0.40, 0.03, 40
BASKET_EXT_M = (0.20, 0.70)
BASKET_RIM_BAND_M = (0.12, 0.27)   # the basket's rim is 0.19 m over the floor (asset)
BASKET_MATCH_M = 0.45          # a basket sighting this close to a track updates it (two tracks 0.3 m apart were one basket, t20f14)
BASKET_REACH_M = 0.45          # a vegetable this close to a basket's estimated centre lies in it (onions at 0.42-0.55 had no group, 301)

# ---- picking --------------------------------------------------------------------------------------------------------------------------
LOOK_STEPS = 18               # controls standing still before a look is judged (detections every control in the look phases)
WRIST_LOOK_STEPS = 10
VEG_STAND_RINGS = (0.55, 0.65, 0.75, 0.9)
N_BEARINGS = 24
MAX_STANDS = 300
VEG_STAND_CLEAR_M = 0.30      # a pick stand this far from the perceived obstacles (its centre; the base radius is 0.28)
SIDE_COST = {'left': 0.0, 'right': 0.15}
FACE_MAX = 1.0                # a pick stand faces its vegetable within this bearing
VEG_PLAN_MAX = 5              # nearest vegetables tried per plan
MAX_PLAN_FAILS = 6
BAD_SPOT_M, BAD_SPOTS_KEPT, BAD_SPOT_STEPS = 0.06, 6, 1500
NO_STAND_STEPS = 300           # a vegetable without a pick stand from one view is planned again after this
VEG_REFIND_M = 0.20           # from the stand the vegetable is looked for among the fresh tracks of its kind this close to the plan
VEG_FOLLOW_M = 0.12           # the vegetable seen again from the stand within this of the plan
REACH_TIMEOUT = 400
GRASP_SPREAD_M = 0.05
APPROACH_SPEED, LIFT_SPEED = 0.003, 0.003
DESCEND_WALK_MAX = 150
DESCEND_SHORT_M = 0.03        # a descent ending farther than this from the pinch is a failure (no close)
HEAD_REST_XZ = (0.33, 1.50)   # the head camera in the robot frame with the trunk at rest (FK)
GRIP_WAIT = 45
HELD_GAP_M = 0.008            # closed fingers this far apart hold something (fully closed: about 0)
HELD_CUT_M = 0.30             # the held vegetable is cut out of the clouds within this of the finger midpoint (a leek is 0.31 m long)
VEG_LIFT_M = 0.15
CARRY_AT = (0.35, 0.15, 1.20)  # the raise ends at this height (clear of the counter edge)
# the carry pose (trunk 4 + arm 7 joints): hand pointing down at (0.40, +-0.25, 1.10) m in the robot frame, jaw along x, torso yaw 0.
# Fixed joints, not an IK goal: a solve_near carry pose turned the torso (trunk joint 3) and the head looked at the next bowl instead of
# the one in front (native t20f4, 302). Solved offline with the yaw locked and the lean within 0.25 rad of the rest; the head then looks
# 13 degrees down from 1.58 m and sees a bowl 0.7 m ahead beside the hand.
CARRY_Q = {'left': (0.25, -0.714, -0.352, 0.0, -1.264, 0.08, -0.791, -0.258, -0.782, 0.104, -1.558),
           'right': (0.25, -0.712, -0.342, 0.0, -1.296, 0.063, -0.94, -0.163, -0.631, 0.068, -1.512)}
CARRY_WALK_MAX = 300
PICK_VERIFY = 8               # cheapest pick candidates checked with the solve_near solution the servo will walk
VEG_OPTIONS = 4               # cheapest pinch options of a vegetable that are scanned
HAND_HALF_M, WALL_MARGIN_M = 0.04, 0.04   # finger spots and hand edges this far inside the basket's estimated outer walls (about 3 cm
# from the inner wall; 3 cm from the true wall caught a rim, oracle)
VALLEY_M = 0.015             # the pinch point's column must reach this close to the slab's top
PINCH_GRID_M, PINCH_LOCAL_M, SLAB_HALF_M = 0.03, 0.06, 0.025   # candidate raster, local-axis radius, half thickness of the slab
RUN_DEPTH_M, RUN_GAP_M = 0.02, 0.012   # the top run: slab points within this of the top, without a gap wider than this
KIND_VALLEY_M = {'green': 0.03}    # broccoli florets and bok choy leaves are bumpy
KIND_RUN_DEPTH_M = {'green': 0.035}
LOW_COST = 3.0                # per metre a pinch lies under the blob's top (a pile is taken from above)
PINCH_APART_M = 0.04
ROUND_KINDS = ('onion',)      # also pinched diagonally (four jaw directions)
MIN_RUN_M = 0.012             # a top run narrower than this is noise (an onion 1-2 m away has 4-6 points across)
KIND_MAX_WIDTH_M = {'corn': 0.07}   # a corn is 0.057 wide: a wider slab spans two
FINGER_COL_R_M, FINGER_CLEAR_M = 0.015, 0.02   # a finger landing spot is blocked by any point this close (xy) and this far over the pinch
LOW_CLOUD_TOP_M = 0.5         # the latest standing views' points under this height are kept for the finger checks
COLUMN_R_M = 0.02             # another vegetable's points this close to a finger column over the vegetable's top cover the pinch
VEG_MAX_WIDTH_M = 0.085
MIN_WIDTH_M = 0.02
KIND_THICK_M = {'corn': 0.057, 'onion': 0.071, 'green': 0.061}   # lying heights of the assets (corn, onion, broccoli)
LEEK_MIN_LEN_M, LEEK_MAX_WIDTH_M, LEEK_THICK_M = 0.20, 0.07, 0.038
PINCH_DEEPER_M = 0.01
FLOOR_CLEAR_M = 0.022
VEG_BACK_M = 0.08
RIM_CLEAR_M = 0.05

# ---- placing (oracle findings, insights/20_vegetables_oracle.md) ------------------------------------------------------------------------
DROP_RINGS = (0.55, 0.62, 0.70, 0.80, 0.88)   # stands by a bowl at these ranges from its centre (0.88 is the oracle's limit; a bowl in the
# inner corner of the counter, bowl_1 of 302, has no free ring point closer)
YAW_OFFSETS = (0., 0.35, -0.35)
DROP_STAND_CLEAR_M = 0.30    # stands by a bowl this far from the perceived obstacles (the base radius is 0.28)
DROP_SOLVES = 14              # stands tried with solve_near and the body check per drop plan, nearest first
DROP_TRIES = 3                # stands tried by the bowl before the vegetable is put down
GROUP_SIZE = {0: 6, 1: 4, 2: 3}   # vegetables per group (BDDL): a bowl without a stand goes to the smallest group
DROP_FAIL_COST = 3.0          # a pick with an arm that found no drop stand by that bowl before costs this much more, per failure
REACH_FRACTIONS = (1., .75, .5, .25)   # points of the joint walk over a bowl that are body-checked (a forearm crossed a shelf edge half way)
MAX_JUMP_RAD = 2.0            # a reach whose joints change more than this is not walked (its wrist turned over on the way, oracle)
DROP_CLEAR_M = 0.03           # the held vegetable's bottom this far over the rim at the end of the walk ...
BOWL_ABOVE_M = 0.12           # ... plus this much (then straight down)
REACH_LOWER = (0., 0.05, 0.10)   # the walk's end is tried this much lower when the arm meets something (wall cabinets)
IN_BOWL_M = 0.06              # lowered until its bottom is this far under the rim (lower, a corn wedged on the walls and pushed the bowl)
SPREAD_M = 0.06               # the second and later vegetables of a bowl go this far off its centre
BOWL_ABOVE_TOL_M = 0.08
RELEASE_WAIT = 30
STOW_OVER_RIM_M = 0.06        # the empty hand rises at least this far over the rim before it goes back
STOW_AT = (0.15, 0.25, 1.20)  # after a release the hand goes straight here (robot frame: x, y to its side, at least this high), then tucks
STOW_MAX, TUCKED_RAD, TUCK_STALL = 300, 0.25, 20
LOWER_CONTACT_M = 0.02        # lowering into a bowl stops and lets go once the hand lags this far (0.03 tilted bowl_3 of 302)
VEG_RETRACT_M, RETRACT_OVER_RIM_M = 0.12, 0.18   # the empty hand rises this far, and this far over the rim, before the arm tucks
PUT_FLOOR_M = 0.20            # finger midpoint height of a put-down
BODY_MARGIN_M, WRIST_ABOVE_MID_M, CEILING_SLACK_M = 0.06, 0.24, 0.03   # the wrist of a hand pointing down under a shelf over a bowl
SHELF_IN_M = 0.20              # under a shelf the walk over a bowl ends this far out, the hand goes in on a straight line (oracle)
CEIL_MARGIN_M, CEIL_MIN_ABOVE_M = 0.06, 0.25   # a shelf is searched over the bowl's rim circle grown by this, from this over the rim
