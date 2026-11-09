# x02: x01 with a last trip to the kitchen side for the bar's battery

Second controller for `store_batteries` (task 82): `x01` (below) with one more phase, the bar's battery from the
kitchen side (see "x02" at the end). Everything else is x01's.

x01 is the gift-bags controller `m02`
(`tasks/make_gift_bags_for_baby_showers/experiments/candidate/m02`, itself the `dispose_of_glass` controller `w01`) in
the same scene, `Rs_int`, with one new skill: **opening a drawer**. The robot first pulls the top drawer of the
cabinet by the living room's west wall open by its handle. Then it takes the batteries from the coffee table and the
bar, two at a time, and lets them fall into the open drawer. The design (map prior, scan-matching localisation,
things found by size over a known surface, a pinch from above, staging points for turning) is described in the w01,
g01 and t01 READMEs; this file lists what differs.

## The task and its score

- **The batteries** (`battery.n.02`, model `dcjyzg`): three, 3.4 cm across, 6.1 cm long, standing.
  - Two stand on the coffee table (top 0.398 m), anywhere on it.
  - One stands on the bar's countertop between the living room and the kitchen (top 1.079 m), anywhere along it.
- **The goal:** some cabinet with all three batteries inside it. Two cabinets are in the task: the one by the west
  wall (`bottom_cabinet_bamfsz_1`, four drawers) and the one by the east wall (`bottom_cabinet_jhymlr_0`, six).
- **The score.** For each cabinet, the share of batteries inside it; Q is the better of the two (1 when all three are
  in one). Each battery in the drawer is worth 1/3.
- **Horizon:** 11556 steps.

## The drawer (new: `drawer.py`)

- **Inside counts in an open drawer.** In a dev job (oracle) a battery let go into each of three drawers opened
  0.12-0.20 m was inside its cabinet. So the drawer is only opened, never shut.
- **The drawer.** The top drawer of `bottom_cabinet_bamfsz_1` runs out along +x, 0.36 m at most. Its handle is a bar
  0.40 m wide, 1.6 cm high, 0.672-0.688 m up, standing 1.8 cm off the front at x -1.529 (from the probe's collision
  meshes).
- **The grasp.**
  - The base stands facing west, the handle 0.59 m ahead and 0.15 m to its left (between the sofa, the floor lamp and
    the coffee table).
  - The left hand comes level from the east, 10 cm straight in, its fingers closing up and down across the bar.
  - It is led 2 cm behind the front: it stops at the front with the bar between the fingers. Aimed at the bar itself,
    from a base 2 cm further back than the map pose had it, the fingers closed in front of the bar (302).
  - **Centring.** If one finger stops much further out than the other, the bar is off the hand's middle. The hand
    opens, moves by half the difference toward that finger, and closes again. Without it the hand closed 1.5-2 cm
    under the bar, and the pull slipped off after 6-8 cm.
  - The planner lets the hand into the cabinet's outline by up to 11 cm: the lower finger, 5 cm under the bar, is
    inside the column the map's height field makes of the cabinet.
- **The pull.** The base backs straight away 0.22 m at 5 cm/s with the hand fixed. The fingers are watched every
  step: if they slip off, the base stops, the hand takes the bar again where it now is, and the pull goes on (up to
  four grasps).
- **Measured** (dev jobs with simulator truth for the base pose, starts 301 and 302): the drawer stood 0.219-0.221 m
  open after one grasp, also with the pose off by ±2.5 cm along the drawer and 2 cm across it. In whole runs it opened
  0.22 m on every start tried.

## What else differs from m02

- **The batteries are looked for** over three surfaces (`SURFACES` in `observer.py`): the coffee table, the bar's
  countertop and the floor. A battery is a patch of points 1.2-9 cm over its surface whose top is 2.5-7.5 cm over it
  and which is 1.5-10 cm across.
  - The leaves of the pot plant on the coffee table hang beyond its outline in the map and gave false batteries:
    - nothing may be over the patch;
    - its lowest points must be within 3 cm of the surface (a battery stands on it; a leaf hangs over it);
    - three spots at the plant's edge (`FALSE_SPOTS`), taken for batteries in several runs, are left out (4 cm
      around each; no battery of starts 301-310 stands there).
  - On the floor only from nearer than 1.8 m (far off the floor's clutter gave 20 false batteries in a run).
- **The pinch** closes 2 cm under the battery's top, as its sighting gave it. The other batteries are boxes of their
  own height for the validator.
- **The drop.** The batteries are let go 3.5 cm above the drawer's top edge, over the middle of its open part, each
  hand 5 cm to its own side. The open drawer is a box for the validator.
- **Stances** (`trash_pipeline.py`):
  - The battery may lie up to 0.86 m ahead of the base (`PINCH_AT`): from the bar's east end the batteries a little
    further west are reached only from 0.79-0.86 m (offline, the arm planner over the map).
  - No camera-led centring in a tight stance (`CENTRE_IN_GAP`). It was made for a can on the floor between the sofa
    and the coffee table; at the table's edge there is furniture on one side only, the measured gap came out
    0.4-0.7 m wide and 20 degrees skewed, and every such stance was given up (309). The base drives in by the map.
  - At most 160 stances are tested with the arm planner per search (`MAX_CHECKS`): the search for a battery out of
    reach took minutes of computation, and a scored episode is stopped after two hours of wall time with no metrics.
  - Only three sideways offsets in a tight place (29 before).
- **No stance in the kitchen** (`KITCHEN_Y`). On the way into the kitchen, through the passage at the bar's east end,
  the map pose went 0.2 m and 14 degrees off, and then 1.5 m (303). So the bar's battery is taken from the living
  room, from the bar's east end, or not at all: the chairs stand along the rest of it.
- **Map** of the scene with the kitchen, without the batteries (`dev/build_scene_map.py --bounds -2.4 -4.4 2.0 3.6`).
- **Lookouts** east of the coffee table and north of it, with the bar in view.

## Known limits

- **The bar's battery** stands west of x -0.9 on starts 301, 304, 306 and 310. There the chairs keep the base away
  and it is not taken: those starts reach at most Q 2/3. (On 302, at x -0.46, it was taken from 0.88 m in the last
  development round.)

## x02: the bar's battery from the kitchen side

- **Why.** On 301, 304, 306 and 310 the bar's battery stands behind the chairs, where x01 has no stance from the
  living room (Q at most 2/3 there).
- **When.** Last: once nothing else is wanted (every other battery is in the drawer or given up after four misses),
  before step 8000, once per episode. Whatever happens in the kitchen, the batteries already in the drawer stay in.
- **How** (`from_the_kitchen` in `trash_pipeline.py`): the base goes to (-0.60, 2.20) in the kitchen, looks, and takes
  the battery from a stance on the kitchen side (the ban on kitchen stances, `KITCHEN_Y`, is lifted for this trip
  only); then back to (0, 0.20) in the living room and into the drawer as usual.
- **No fresh global fix in the kitchen.** It was tried: the kitchen's scans match several places, and the global
  localiser took a pose 2 m and 35 degrees off where the tracked pose was right to 1.5 cm (304, 310, dev round k2).
- **Only east of x -1.20** (`KITCHEN_WEST`). West of the oven (301: x -1.56, 306: x -1.65) it keeps the base away on
  the kitchen side too, and on the way to the one at x -1.28 (304) the robot fell over in the passage (round k3).
  So the trip is made on 310 (x -0.99) of the public starts; the others behave as x01.
- **Development** (round k3, dev server): on 310 the battery was taken from the kitchen side (map pose right to
  1.8 cm) and let fall into the drawer: all three in, success at step 5044+.
