# w01: disposing of water glasses with the trash controller

First controller for `dispose_of_glass` (task 76). It is the `clean_up_broken_glass` controller `g01`
(`tasks/clean_up_broken_glass/experiments/candidate/g01`), itself the `picking_up_trash` controller `t01`
fitted to another scene. Here the things to take stand on a sink instead of lying on the floor. The design is
described in the t01 and g01 READMEs:
- a map prior and scan-matching localisation;
- things found by size where the map has nothing;
- one thing per hand, and staging points for turning;
- the trash can found by its rim and approached with its middle 0.58 m ahead.

This file lists what differs. The module and variable names still say "can" and "soda": here they stand for a
water glass.

## The task

- **Scene:** `hotel_suite_large`. The robot starts in the bathroom.
- **The glasses:** four water glasses (`water_glass.n.02_1..4`, model `evaida`), 8.4 cm across, 15.7 cm high,
  0.1 kg. They stand on the top of the bathroom's wall-mounted sink, a counter 2.6 m long (x 1.82-2.33,
  y 2.35-4.92) at 0.79-0.81 m, 0.07-0.32 m behind its front edge.
- **The trash can:** the same open ashcan as in the restaurant (`cjmezk`, rim at 0.42 m, 0.38 m across). It
  stands somewhere in the bedroom, south of the bathroom.
- **The goal:** each glass inside the trash can. Q is the fraction of the four goal predicates met. The native
  horizon is about 13800 steps (1.5 x the human mean of 9195).

## What is different from g01

- **Where glasses are looked for.** Over the sink's top only (`SINK_BOX`), at 0.83-1.02 m, where the point is
  4 cm or more above what the map has there (the faucets are in the map). A glass is a cluster whose top is at
  0.91-1.0 m; its axis is the middle of its rim as the head camera sees it.
- **The pinch.** A glass is pinched by its wall, on the side nearest the robot: one finger inside, one outside,
  closing along the line from the robot, with the fingertips at 0.93 m (2-4 cm below the rim). The fingers
  are 10 cm apart when open, so an error of 3 cm either way still puts one inside.
  - The hand comes straight down the last 13 cm.
  - Until it is above the glass, the glass is a box for the validator. Without that box, the open fingers tipped
    glasses over on the way.
  - The other glasses are boxes from the sink's top to 0.99 m while a hand works near them.
- **The stance at the sink.** The base keeps 0.40 m from the counter, so a glass is 0.48-0.76 m ahead when it is
  pinched (`PINCH_AT`). With the trunk upright the arm reaches it with little load: an offline check found 22-42
  arm poses per stance.
- **What the chassis may go under.** The wall shelf west of the sink's south end (0.51-0.66 m high) and the sink
  itself are in the map's chassis layer (`chassis_also` in its meta). Below the shelf, a stance for a "tight
  gap" put the base where the torso met the shelf, and from there no route was found (start 308).
- **Letting go.** The hand stays pointing down and the glass hangs below it, 8 cm above the rim, over the
  trash can's middle.
- **No keep-out discs around glasses:** they stand on the sink, out of the chassis's way.
- **Lookouts.**
  - For glasses: two places in front of the sink.
  - For the trash can: three places in the bedroom (north of the beds, the strip east of them, its south end).
