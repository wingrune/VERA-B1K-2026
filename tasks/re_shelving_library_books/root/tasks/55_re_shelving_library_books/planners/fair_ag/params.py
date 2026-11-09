"""Constants and plan switches of the fair planner of task 55 (moved verbatim from logic.py). The phase modules import them from here."""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from core.skills_lib import r1pro_kinematics as K

from . import geometry as G
from . import ik

ROBOT_RADIUS = 0.31       # the base footprint radius is 0.28
GRID_RES = 0.05
NAV_GAIN, NAV_VFAR, NAV_VNEAR, NAV_YAW_GAIN, NAV_WMAX = 0.9, 0.5, 0.18, 0.8, 0.5
NAV_POS_TOL, NAV_YAW_TOL = 0.05, 0.08
NAV_REPLAN_EVERY, NAV_STUCK_STEPS, NAV_MAX_STUCK = 90, 120, 3
NAV_TIMEOUT = 1500
SETTLE_STEPS = 12
REST_STEP = 0.02
REACH_POS_TOL, REACH_ORI_TOL = 0.02, 0.30
WAYPOINT_TIMEOUT = 600
RINGS = (0.55, 0.65, 0.75, 0.85)
BEARING_STEP = math.radians(20)
CAND_LIMIT = 8
CAND_LIMIT_PUSH = 30
IK_BUDGET = 8          # first bases of the path-cost ring, both arms; a full ring of 30 never returns offline
GRIP_OPEN, GRIP_CLOSED = K.GRIPPER_OPEN, K.GRIPPER_CLOSED
PUSH_SPREAD_M = 0.046
LIFT_M = 0.10
LIFT_CHECK_M = 0.03
CLOSE_DWELL = 90
RELEASE_DWELL = 40
STEADY_STEPS = 45            # controls the base stays still after a pinch, so the friction grasp can settle before the carry
MAX_ATTEMPTS = 3
SLIDE = dict(down=0.003, push=0.0015, advance=0.002, lift=0.001, retract=0.002, insert=0.0012, lower=0.001)
SHELF_CLEAR_M = 0.012
INSERT_REACH_M = 0.16        # the held fingertip is at least this far in front of the base along the insert slide
HELD_MAX_M = 0.16            # a pinched book's centre is within this of the finger midpoint (0.035 pinch depth plus half the book)
HOLD_JOINT_STEP = 0.006      # rad per control of the joint servo while a book is in the gripper (the free arm uses 0.016)
SOFT_CATEGORIES = ('armchair', 'coffee_table')   # loose furniture the base may push when no route exists around it
CARRY_V, CARRY_W, CARRY_SLEW = 0.18, 0.12, 0.008     # m/s, rad/s and the largest change of a base command per control while a book is carried
CARRY_YAW_FIRST = 0.25       # rad: above this the base turns in place (no translation) so the book is not flung out of the pinch
CARRY_TIMEOUT = 2600
DRIFT_BRAKE_M, DRIFT_FAIL_M = 0.022, 0.07   # robot-frame shift of the book in the hand: stop the base, then give up the grasp
# switches of the plan (ik.variant(), one change each; the fair default is DEFAULT_VARIANT of ik.py)
#   putback a held book with no insert pose is slid back over the desk and released there (once per book), then pinched again
#           from stands within 40 deg of the edge normal
#   aware   a pinch stand is kept only when the book will sit ahead of the fingers along the reach (the insert slides it that way)
#   guard   a push stops once the book overhangs another desk edge 0.02 m more than at the start (and never closer than 0.03 m inside)
#   step10  pushes are at most 0.10 m; the next push is planned from the live pose
#   normal  a book skewed more than 28 deg from the edge is pushed along one of its face normals (pusher on the opposite face centre, push
#           line through the book centre); the travel is cut where the footprint would come within DESK_MARGIN_M of another desk edge
#   centre  the pusher contact sits where the push line through the book centre meets the rear face (no turning moment)
#   safe    a push stops once the footprint gains 0.005 m towards any other desk edge it is inside of (live pose), and backs off along the
#           push before lifting
#   sweep   a stand and arm whose way from the carry posture to the first waypoint sweeps the forearm or gripper through what stands on
#           the desk (the lamp) is not used
#   turn    a turn in place with a book in the hand goes the way whose arc keeps the book off what stands on the desk (backs away first
#           when both ways meet it)
#   turnback  as turn, but when the short way meets it the base first backs straight away until the short arc is clear (the long way
#           round only after 300 controls of backing)
#   land    the pusher is shifted along the rear face so its fingertips do not land on another book (a neighbour pressed against the face)
#   unbrake a carry braked 90 controls by a shift of the book in the hand, with the book still held, takes the shifted offset as the grasp
#   sp1, sp2, sp4  a push towards an edge costs w * (1 + n.y) / 2 more (w = 1, 2, 4): the pinch from the south edge reaches north, along
#           the insert slide, so the book sits ahead of the fingers; the north edge leaves it behind them
#   sforce  when the hand box at the west edge is blocked, a push towards the south edge is taken whenever one is reachable
# BIMBO_55_FAIR (ik.fair_mode), one grasp change per job; empty keeps the pinch above:
#   halt  a push stops when the book-top cloud overhangs by OVERHANG_M (hand travel of the planned distance is the cap)
#   under the pinch midpoint is placed under that visible overhang, at mid height, and a short lip bite is refused
#   jaws  the close waits until the midpoint is at that under-lip pose (it does not close on the desk lip)
#   nest  halt + under + jaws, plus sforce (the chair blocks the west edge; the scoring oracle pinches from the south)
#   tuck  nest, but the approach starts below the lip so the lower finger threads under the overhang
#   m10, m15, m20  onbook, and a pinch past the visible lip is kept when the hand box meets only the desk's own collision margin
#          (the outward occupied run from the collision face ends within 0.10, 0.15 or 0.20 m)
VARIANT = ik.variant()
MARGIN_TOL = {'m10': 0.10, 'm15': 0.15, 'm20': 0.20, 'd08': 0.10, 'd09': 0.10, 'd10': 0.10}
VISIBLE_DEPTH = {'d08': 0.08, 'd09': 0.09, 'd10': 0.10}
ALL_EDGES = dict(G.EDGES, E=np.array([1., 0., 0.]))
NORMAL_SKEW = math.radians(28)
Waypoint = Dict[str, Any]
