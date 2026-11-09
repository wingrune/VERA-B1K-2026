# m03: the toy dice into the gift bags, then the bags onto the coffee table

Second controller for `make_gift_bags_for_baby_showers` (task 86): `m02` with a new first phase. Before the bags are
moved, the robot takes the two toy dice from the top of the cabinet by the east wall, one at a time, and lets each
fall into its own bag where the bags stand on the breakfast table. Then m02 runs unchanged: the bags onto the coffee
table. The small-thing skills come from the `store_batteries` controller `x01` (finding a small thing on a known
surface, a pinch from above at the height its sighting gave); see "The dice (m03)" below. The wafers are not handled.

m02 itself, below, is the `dispose_of_glass` controller `w01` (`tasks/dispose_of_glass/experiments/candidate/w01`) in
scene `Rs_int`. It finds the two paper bags on the breakfast table, takes each from above and sets them down side by
side on the coffee table. The design (map prior, scan-matching localisation, things found by size over a known surface, a pinch from
above, staging points for turning) is described in the w01, g01 and t01 READMEs; this file lists what differs.

## The task and its score

- **The paper bags** (`sack.n.01`, model `bzsxgw`): two, 12-16 cm across, 24 cm high with their handles, 21 g. They
  stand on the breakfast table (top 0.76 m).
- **The toy dice:** two, 4 cm, on top of a bottom cabinet.
- **The wafers:** two, inside the cabinet.
- **The goal:**
  - both bags on the coffee table (top 0.40 m);
  - each die in its own bag;
  - each wafer in its own bag.
- **The score.** Six conditions, none met at the start; each is worth 1/6. m02 reaches at most 2/6 = 0.33.
- **Horizon:** about 14480 steps (1.5 x the human mean of 9651).

## The dice (m03)

- **What counts.** Each die inside its own bag (one die per bag). In a dev job (oracle, `bagin301`) a die let fall
  into a bag from above its opening was inside it. A wafer was not: 20 cm long, it stays across the bag's top. The
  wafers lie in a drawer of the same cabinet and are not handled.
- **The dice** (`toy_dice.n.01`, 3.3-4 cm) stand on the top of `bottom_cabinet_jhymlr_0` (0.754 m), on all of
  starts 301-310 within 15 cm of its front at x 1.306.
- **Finding them** (`Observer.small_things`): from two views in front of the cabinet, facing east, a patch of points
  over its top whose top is 2-5 cm over it, 2-8 cm across, standing on it and with nothing over it.
- **Taking one** (`take_die`): a stance facing east with the die 0.50-0.70 m ahead and up to 0.14 m to the side
  (`DIE_AT`), tested with the arm planner; a look again from there; a pinch from above, the fingertips' centre 1.5 cm
  under the die's top.
- **Into a bag** (`die_into_bag`): a stance found as for taking a bag (with the sideways offsets of a tight place,
  the bag up to 0.82 m away, `BAG_AT`); the hand, fingers down, 1.08 m up over the bag's middle as last seen
  (the bag's rim is at 0.975 m), opens. Where no stance is found, the die is let go where the robot stands.
- **Carrying the bags.** In development a die stayed in its bag while m02 carried the bag by its handles.
- **Development** (round a1, starts 301-308, dev server): both dice taken on all eight starts; both let go into the
  bags on 303 and 308, one on the other six (305: no stance at its second bag).

## What is different from w01

- **The bags** are looked for over the breakfast table at 0.80-1.04 m. A bag is a cluster whose top (its handles) is
  0.95-1.02 m up.
- **The pinch** changes from one try to the next (`RIM_TRIES`, `GRIP_TRIES` in `trash_skills.py`):
  - first by the handles over the bag's middle, 4 cm below their top;
  - then at the bag's own top edge, 0.90 m up, 5.5 cm towards or away from the robot, one finger inside.
  - The fingers close along or across the way to the bag.
  - With the handles alone most pinches closed on nothing: the handles stand over the bag's long sides.
  - With a pinch at the wall from 0.975 m, the fingers came down on the handles and tipped the bags over.
- **The coffee table is fixed** (`TABLE_BOX`, `COFFEE_SPOT`). A bag is let go 3 cm above its top. Each hand sets its
  bag 0.10 m to its own side, and the table is a box for the validator.
- **Map** of the scene, without the bags, dice and wafers (`dev/build_scene_map.py --bounds -2.4 -4.4 2.3 1.9`).
- **Lookouts** west and south-west of the breakfast table.

## Known limits

- **The wafers** (in the cabinet's top drawer) are not handled: they do not fit through a bag's opening.
- **A bag with no stance** near it on the breakfast table (chairs stand around it) gets no die.
- **The coffee table's place** was measured on start 301.
