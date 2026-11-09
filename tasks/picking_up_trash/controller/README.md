# t01: picking up trash with a map prior and classical motion planning

First controller for this task. It follows the design of the fax and battery controllers (`m05`, `b01`)
and reuses their whole-body IK, joint-path planner, base driving and the frame of the localiser.

## The task

Three soda cans (7.6 cm across, 12.7 or 13.5 cm tall, 0.35 kg) stand on the living-room floor of
`house_double_floor_lower`. They must end up inside the trash can, which stands somewhere on the
kitchen floor, where the robot also starts. Kitchen and living room are one open space; the cans are
1 to 7.5 m from the trash can. Horizon 7901 steps (1.5 x the human mean of 5268). Each can inside is a
third of the Q-score.

## What it does

1. **Localise** in the kitchen against a static map (`priors/house_map.npz`), with a full turn.
2. **Look around**: cans and the trash can are things of their size standing on floor the map has
   empty. What the first turn shows is taken; cans that come into view later are added on the way.
3. **Fetch** a can with each hand: drive to a point from which the stance is a straight move, turn
   there, move in, bend down and pinch the can from above, fold the arm, move back out.
4. **Deliver**: drive to the trash can the same way and let the cans go above it, lying.
5. **Repeat** for the third can. If cans are missing, look around from three places in the living room.

An episode takes 2800 to 5000 steps of the 7901 on most starts (median 3500); three cans standing in a
cluster took 7300 in a development run.

## Why it is built this way

- **The floor is within reach, and the simulator lets the trunk carry it.** With the can 0.37 m ahead
  and 0.13 m to the side of the arm, the fingertips reach 0.095 m above the floor with the trunk folded
  over the base. Statically that loads the first trunk joint with 1.2-1.8 times its effort limit; the
  joint follows anyway (fingertips within 4 mm of the target in every trial, `dev/jobs/floor_trial.py`,
  12 of 12 cans taken and delivered). Of the poses the IK finds, the least loaded is used.
- **Cans are let go lying, crossed.** A can counts as inside when the middle of its box is within the
  trash can's fillable volume, which ends 0.22-0.24 m above the floor (the rim is at 0.28 m), and the
  trash can narrows to 0.065 m inner radius at the bottom: three cans do not stand side by side in
  it, they pile up. Let go upright over the middle, the top one's middle ended at 0.17-0.25 m and
  missed in 3 of 14 trials; spread 4.5 cm apart it still ended as high as 0.23 m and missed in a run.
  Let go lying, each across the others, it ended at 0.13-0.19 m in 41 of 42 trials and at 0.22 m in
  one, and all 42 counted (`dev/jobs/drop_study.py`). The hand that pinched a standing can from above turns level for this;
  each hand has a direction it holds a can in most easily (60 degrees off straight ahead, to its own
  side), so two cans from one stance cross, and the third is laid across both.
- **Turning happens only away from the cans.** The chassis is a rectangle whose corners reach 0.42 m
  in front and 0.53 m behind the base centre. A turn next to a can sweeps it, and a route that squeezes
  between two cans half a metre apart brushes one: both knocked cans over (two of twenty public starts
  in one sweep). So a stance comes with a staging point that lies 0.56 m from the surface of every can
  and of the trash can; the planned route ends there, the base turns there, and the rest is a straight
  move in and the same way back. Routes keep 0.6 m from every can that is known (0.68 m from the trash
  can) and are planned again when a new one comes into view. The order of the cans is chosen with the
  same rule, so a can that stands in a passage is taken before the ones behind it.
- **Where that leaves no way, the base passes nose first.** A trash can that stands a metre from a chair
  shuts half the kitchen off for a base that wants room to turn (training start 284: no route at all).
  The chassis is 0.69 m wide whichever way it drives, but only along its heading does it stay that
  narrow: such a way is planned with the cans and the trash can as wide as that needs, checked with the
  chassis's rectangle driven along it nose first (5 cm to spare), and driven with the heading kept along
  the way, standing still while it turns at a bend. If even that finds no way, it is tried once more
  with the base centre 0.38 m from furniture instead of 0.42 m (3 cm between chassis and furniture).
- **A stance is checked with the arm before the base goes there.** Next to the fireplace and the walls
  the arm has no way down from some sides, and cans that stand 0.4 m apart are on each other's stances.
  The check has the other cans as boxes in the validator, as the pinch itself will.
  If a can has no stance for one hand, the other hand or another can is tried first. Sixteen headings
  round the target are tried, then forty-eight: a trash can between the fridge and a chair has one or
  two stances, which sixteen headings counted from where the robot happens to stand can miss (training
  start 128). There each hand may also get a stance of its own, and a hand keeps its can until it has
  been let go, over several trips if need be.
- **Nothing folds through the trash can.** It weighs half a kilogram and is not in the map. While an arm
  works above it, it is a box in the collision validator, and it stays one until the arm is folded; the
  open hand first goes straight up 12 cm. (Before that an arm folded from just above the rim, tipped the
  trash can over with two cans in it, and the third can was let go where the trash can had been: public
  start 304.) The other cans are boxes in the same way while a can is pinched and the arm folds.
- **Two cans that touch.** Nine of the 300 training starts have two cans closer than 0.16 m (none of the
  public ones). The detector splits such a pair by the points of the two tops, and the fingers come down
  on the two sides of the line through both cans, where there is room even if they touch.
- **A can that has fallen over is taken lying.** Its top is 0.076 m up; from above its points form a
  strip whose long direction is its axis. The fingers come down across it at the height of its axis
  (`dev/jobs/lying_trial.py`: 7 of 7, direction read to 0.5-3 degrees, once 14). Held this way it
  already lies in the hand and is let go with the hand pointing down.
- **The gap between the sofa and the coffee table** is 0.72 m wide at chassis height; the chassis is
  0.68 m. One of the 960 cans of the distributed starts stands in it, on a public start (317), and the
  table hides it from everywhere but the gap's south end. The route planner takes the chassis for a
  disc and finds no room there, so stances are also tried with the chassis's own rectangle on a layer
  of the map at chassis height. Such a stance is entered straight from a metre outside. The map and the
  map pose are each good to about the 2 cm there are on each side, and a heading 1.7 degrees off costs as
  much again over the chassis's length; so in front of the gap the base lines itself up with what the
  head camera shows of the two sides (a line fitted to each, `Observer.gap_sides`): middle and
  direction. One lookout of the search stands at the gap's south end.
- **Localisation in height slabs.** The scan match of the earlier tasks compared one band of the depth
  image (0.25-0.90 m) with one layer of the map. In the house a cabinet 0.41 m high is a block in that
  layer, while the camera sees the wall behind it above 0.41 m: with true poses the match was off by
  8 cm (median) and up to 22 cm, and a run started 9 cm off. The map now has four thin slabs that lie
  between the heights where furniture ends (0.22-0.38, 0.46-0.64, 0.95-1.25, 1.40-1.85 m), and the scan
  is cut the same way: 1.4 cm median, 2.9 cm at most on the same views; 1.5-2 cm median along runs
  (`dev/study_sight.py`, `dev/test_localizer.py`). Movable furniture is in these layers: no instance
  of the task moves it.
- **Where a can is, for the hand.** The head camera looks down on a can: the points of its top lie
  around its axis (3-6 mm from a metre, `dev/test_floor_things.py`; the trash can, by a circle fitted
  to its rim, 2-7 mm at any range). From the stance the can is below the image, so the last sighting
  from the way in is carried along by dead reckoning from the velocity command, which over the last
  half metre is good to a few millimetres whatever the map pose does: over 116 pinches of the last
  development sweeps the hand's target was 6 mm off at the median and 9 mm at most. Over a long way with
  turns it is not: once the sighting was 700 steps old, the target was 3 cm off and the fingers knocked
  the can over. So if, when the base is ready to move in, it has driven more than 1.5 m or turned more
  than 2 rad since it last saw the can (from a staging point to the side the can is not in the image),
  it faces the can, looks, and turns back; if the can is then nearer than 0.8 m (the image's lower edge
  cuts it), it first steps straight back to 0.85 m where there is room. The same if its last sighting
  was from more than 1.05 m: on
  recorded runs with true poses a can is placed to 5 mm (median; 9-22 mm at most) from 0.65-0.9 m, and
  to 10 mm (up to 35 mm) from 1.1-1.6 m. Of the sightings the nearest is kept while it is fresh.
- **What the hand finds decides what is known.** A pinch that closes on nothing means the can is not
  where it was believed: it is forgotten until it is seen again. A can seen lying where one stood is
  that can, knocked over, not a second one.
- **False cans.** Chair and table legs are of a can's width: things are looked for at least 0.10 m from
  anything in the map (no can of the distributed starts stands closer than 0.22 m to furniture, the
  trash can no closer than 0.16 m), cans only on the living-room floor and the trash can only on the
  kitchen floor, as the task places them. A can must be seen twice to count and is forgotten when six
  looks at its place do not show it.

## Inputs and priors

Runtime inputs: head RGB-D (depth only is used), 61-D proprioception, the cameras' poses relative to the
base, the policy's own last action. The wrist cameras are not used.

Priors, the same for every instance: the robot model; the map of the house's lower floor, built once
from the scene's collision geometry (`dev/build_scene_map.py`); the sizes of a soda can and of the trash
can; which floor the task puts each on. In the source besides: the three lookouts of the living room and
the two of the kitchen, the heights of the localisation slabs, the odometry model measured for the
battery task.

No simulator state, segmentation, contacts or evaluator results are read.

## Files

| File | Role |
|---|---|
| `observation_policy.py` | Evaluator entry point (`Policy`) |
| `trash_observed.py` | Estimator fed by the cameras (tracks of cans, the trash can), per-step runtime, looking around |
| `trash_pipeline.py` | The task as a sequence of skills, driven by an estimator (camera-based or, in dev jobs, simulator truth) |
| `trash_skills.py` | Floor pinch, fold, release above the trash can |
| `observer.py` | Cans and the trash can in the depth image, the sides of a gap, localisation scans |
| `manip.py` | Context with the collision validator, base driving, joint-path planning |
| `localizer.py`, `scene_map.py` | Scan matching in height slabs, dead-reckoning corrections; map prior, routes, clearances |
| `wbik.py`, `skills.py`, `kinematics.py`, `robot_collision.py` | Whole-body IK and joint-motion primitives (from m05) |

## Known limits

- A can that leans against something (neither standing nor lying) is not recognised.
- Nothing checks that a can that was let go landed in the trash can; one that bounced out is found only
  if it lies on the living-room floor or within sight on the way (the search for cans covers the
  living room).
- The trash can is recognised by its rim; knocked over it is not found again.
- A can in a hand is not in the collision model: the folded arm carries it clear of everything in the
  runs made, but nothing proves it. An arm that is held up on its way to the trash can folds again and
  tries another direction (it happened once, with a can pinched lying, next to the stairs).
- If the trash can is knocked over, its place is not found again and what is let go afterwards misses.
- Cans closer than about 0.4 m to each other from all sides wait for the outer ones to be taken; a can
  whose every stance stays blocked is left after two rounds.
- **The trash can is never moved.** Where it stands in the only passage out of the part of the kitchen
  the robot starts in (with the table and its chairs, it shuts the robot in on 8 of 279 starts outside
  the held-out list, none of them public), the base passes it nose first if 0.69 m of chassis fit, and
  on about half of those starts they do not: the episode then ends with nothing delivered. The battery
  controller b01 carries its trash can out of the way; this one does not.
- The gap between the sofa and the table is entered with 2 cm to spare on each side; the table is
  movable and gives way, the sofa does not.
- The map is of asset version 3.9.0; furniture that another version puts elsewhere would mislead both
  the localiser and the detector.
