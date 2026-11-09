"""The kitchen of cleaning_up_plates_and_food in numbers (map frame M, metres), measured in this task's scene
(dev/jobs/probe_static.py, probe_goal.py, plates_trial.py, door_trial.py). The same in every start: only where the
plates, pizzas and bowls stand on the table and where the robot starts change.
"""
import numpy as np

# ---- the breakfast table: a top 0.646-0.679 m up on four legs; the chassis goes in under the top's edge ----------
TABLE = dict(x=(3.826, 4.756), y=(.693, 2.183), top=.679)
# Its long sides have chairs (they can be pushed, but are not); the hands work from its two ends, the base facing
# along the table: south end (the fridge 0.74 m behind the table's edge) and north end.
ENDS = dict(south=dict(y=.42, yaw=np.pi / 2), north=dict(y=2.45, yaw=-np.pi / 2))
END_X = (3.70, 4.90)                    # the base's middle along an end

# ---- the things on it ---------------------------------------------------------------------------------------
PLATE = dict(radius=.204, height=.024, mass=.18)        # white, 0.408 m across; a pizza on it (0.30 across, 0.83 kg)
BOWL = dict(radius=.075, height=.059)                   # white, 0.15 m across, open at the top
# Where the fingers' ends go on the rim's near side, seen from the base: height above the table's top, and how far in
# from the rim's outer edge. Held (assisted grasp) in trials at the pitches of GRIP_PITCH: bowl 3 of 3, plate 3 of 3,
# the pizza staying on the plate (it leans 0-5 degrees).
GRIP = dict(bowl=(.050, .004), plate=(.016, .008))
GRIP_PITCH = dict(bowl=(90., 70., 55.), plate=(90., 75.))

# ---- the sink: a drop-in sink in the counter along the east wall; a bowl let go over its right-hand basin falls in
SINK = dict(basin=((7.869, 8.309), (.227, .558)), over=(8.09, .39, 1.00), stance=(7.40, .39, 0.), staging=(6.95, .39))

# ---- the fridge (fridge_petcxr_0): side by side, the west half (behind its right door) has the shelves a plate fits
# on (0.52 m wide, 0.46 deep); the east half (0.39 m wide) does not take a plate.
HINGE = np.array([4.679, -.181])        # the right door's hinge (its link's origin)
HANDLE = np.array([5.218, -.047])       # the handle's front face, door shut; the bar runs from 0.71 to 1.69 m up
HANDLE_Z = 1.20                         # where the hand takes it (the bar is 4.7 cm deep here, its foot included)
DOOR_LENGTH = .58                       # from the hinge to the door's free edge
DOOR_OPEN_FROM = .12                    # the fridge counts as open with the door further open than this (rad; 0.1 shut, 0.2 open)
SHELVES = (.969, 1.314)                 # tops of the two lower shelves of the west half (y from -0.26 to -0.715)
STACK = .046                            # a pizza's height on its plate: the second plate is set down on the first one's pizza
PLATE_AT = (4.96, -.44)                 # where a plate's middle goes on a shelf: its pizza's middle is then inside the
                                        # fridge's container volume (x 4.735-5.19, y -0.579 to -0.263) even 10 cm off the plate's middle
