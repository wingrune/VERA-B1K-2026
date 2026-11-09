# f01: cleaning up the plates and the food with a map prior and classical motion planning

First controller for this task. It starts from the controller for `putting_away_Halloween_decorations` (`h01`) and,
through it, from `p01`/`c01`/`u01`: the same localisation, routes, whole-body IK, arm planner and pinches. What is
new is the kitchen: things taken from a breakfast table, bowls let go into a sink, the fridge's door opened and shut
by a hand, and plates with pizzas on them set onto a fridge shelf.

## The task

The robot starts somewhere in the kitchen and living room of `house_double_floor_lower`. On the breakfast table
stand two bowls and two plates, each plate with a pizza on it. At the end each pizza must be on a plate and both
pizzas inside the fridge, both bowls inside the sink, and the fridge not open. Horizon 20544 steps.

What the simulator makes of this (`dev/jobs/probe_static.py`, `probe_goal.py`, `plates_trial.py`, `door_trial.py`,
`door_stages.py`; numbers in `kitchen.py`):

- The table's top is 0.68 m up. A plate is 0.408 m across, a bowl 0.15 m; the gripper opens 10 cm, so both are
  pinched by the rim. A bowl let go over the right-hand basin of the sink falls in and counts as inside.
- The fridge `fridge_petcxr_0` is a side-by-side: only its west half (behind the right door) has shelves a plate fits
  on (0.52 m wide, 0.46 deep); the second shelf is out of the arm's reach from where the base can stand, so the
  second plate goes onto the first one's pizza. A pizza counts as inside when the middle of its bounding box is in the
  fridge's container volume.
- The right door turns about a hinge at its west edge from 0 to 2.36 rad and stays where it is let go; the fridge
  counts as open beyond 0.118 rad. Its free end is a frame 7.1 cm thick; on its outer side the handle bar stands 2.5
  cm off the face from 0.71 to 1.69 m up; on its inner side are bins.

## What it does

1. **Localise** and **find the four things** on the table from the head camera's colour and depth images (white
   rings at the table's height; a pizza's orange inside a plate's ring; one thing seen twice in one look is kept
   once).
2. **The bowls** into the sink: each taken by its rim from a table end, carried, let go over the basin.
3. **A plate that a hand reaches only from the table's south end** is moved to the north half first (its rim may
   stand a few centimetres over the table's edge; if there is no room there it is left where it is): the open door
   will shut the way into the gap between the table and the fridge.
4. **The door** is led open by its handle to 105 degrees from a stance east of it.
5. **Each plate**, taken by its rim with the fingers 75 degrees down (so the arm reaches in to the shelf's depth at
   both heights), is drawn in slowly and carried level; the localiser keeps correcting the pose with the trunk bent
   over the plate. At the fridge the base faces west east of the opening; the arm planner brings the plate to the
   opening keeping its rim off the open door (as balls round the rim); a look at the plate puts its place in the hand
   right; it is slid in along a straight way (switching to the other branch of the arm's solutions if the first
   cannot go the whole way), lowered and let go; the hand is drawn back by a way that keeps off the door, and the
   closed hand pushes the plate in by its near rim to the shelf's depth.
6. **The door is shut**: from a look at it, the hand pinches the free edge across its thickness (again 3 or 6 cm
   further in along the door when the fingers closed on nothing: the pose stands 3-6 cm off by the fridge) and leads
   it to 17 degrees (from beyond 80 degrees first to 62 from a stance 0.2 m nearer the door: from the usual one the
   free end stood at the arm's reach, and the hand lost it); then takes the handle and leads it to shut; then the closed hand presses the outer face (or
   presses at once if the handle was missed with the door nearly shut). Tried up to five times, each from a fresh
   look at the door.

## Files

`kitchen.py` (the kitchen's numbers), `table_view.py` (detector of the things on the table), `plates_observed.py`
(estimator: things, looks at the door and at a held plate), `door.py` (the door's stages: lead by handle or edge,
push, press), `plates_pipeline.py` (the task), `observation_policy.py` (entry for the native evaluator via
`run_native.py`); the rest is shared with `h01`/`p01`/`c01`/`u01`.
