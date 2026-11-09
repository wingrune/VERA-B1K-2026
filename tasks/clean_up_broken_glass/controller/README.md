# g01: cleaning up broken glass with the trash controller

First controller for `clean_up_broken_glass` (task 80). It is the `picking_up_trash` controller `t01`
(`tasks/picking_up_trash/experiments/candidate/t01`) with the sizes, the map and the search places of this
scene. The design (map prior, scan-matching localisation, things found by size on floor the map has empty,
top-down pinch from the floor with the trunk folded over the base, one piece per hand, staging points for
turning) is described in the t01 README; this file lists what is different. The module and variable names
still say "can" and "soda": in this task they stand for a piece of glass.

## The task

- **Scene:** `restaurant_diner`. The robot starts in the dining room.
- **The glass:** three pieces of broken glass (`broken__glass.n.01_1..3`), 5 to 13 cm across, about 4.5 cm
  high, 0.07 kg, on the dining-room floor. Over the public starts they lie at y from -7.9 to 2.54.
- **The trash can:** an open, flared ashcan (`ashcan`, model `cjmezk`) with its rim 0.42 m above the floor
  and about 0.38 m across at the top. It stands in the short corridor north of the dining room (floor
  x -4.03..-2.07, y 3.87..4.75), anywhere along it: x -3.54 to -2.46 over the public starts. The corridor's
  whole south side is open to the dining room.
- **The goal:** each piece inside the trash can. Q is the fraction of the three goal predicates met.

## What is different from t01

- **Sizes.** A piece is pinched at 3 cm above the floor; the observer takes things 4.0-5.0 cm high on empty
  floor as glass and a thing with a 0.38 m rim at 0.42 m as the trash can (`observer.py`,
  `trash_skills.py`). Things up to 0.47 m are looked for, so that the rim is in the band.
- **Map** of this scene (`priors/house_map.npz`, built by `dev/build_scene_map.py` without the floors,
  ceilings, glass and trash can). The robot localises on the dining-room floor.
- **Where it looks.**
  - For glass: six places in the dining room and the bar (`LOOKOUTS['cans']` in `trash_observed.py`); a piece
    is 4.5 cm high and shows from 2.5 m at most.
  - For the trash can: three places in front of the corridor's open side.
- **What is not glass.** The sills and chair feet at the corridor's mouth and along the dining room's west
  wall looked like pieces of glass. Sightings north of y 2.9, or west of x -3.95 and north of y -0.2, and
  sightings within 0.8 m of the trash can are not taken (`GLASS_NORTH`, `GLASS_WEST`, `BIN_ZONE`).
- **Searching.** When nothing known is left, the lookouts are visited again, up to six rounds, until three
  pieces are in the trash can. t01 stopped after two rounds, and after any round that showed nothing new.
  A round ends at the first lookout that shows a piece not known before. Pieces that both hands failed to
  pinch stay known, and in t01 they ended every round at the nearest lookout.
- **The stance at the trash can.** The base stands with the trash can's middle 0.58 m ahead (t01: 0.47 m),
  or else 0.55 or 0.64 m. While a hand is over it, the trash can is a box in the validator: 0.38 m across,
  from the floor to the rim. With its middle 0.49 m ahead, the front of the chassis is inside that box, and
  the planner found no arm pose at all (an offline check with the stances of start 306: none at 0.49 m,
  15-24 per direction from 0.55 m on). In development this left both pieces in the hands on several starts.
- **Letting go.** A piece is let go 8 cm above the rim; where no arm pose is found, 8 cm higher still.
- **One piece, one thing.** A piece up to 13 cm long can show as two things a few centimetres apart; the
  second was then in the way of the hand pinching the first. Things within 8 cm of each other count as one
  (`DUPLICATE` in `trash_pipeline.py`).
- **The trash can's rim** is taken when the rim points lie on average within 3 cm of a 0.38 m circle (t01:
  2 cm). On start 309 the fit was 2.0-2.2 cm and the trash can was never found.
- **Lookouts that a piece shuts.** A seen piece keeps the base 0.6 m away. Where that shut the only passage
  to a lookout (start 305), the route is planned again without it: the base may pass over a piece.

## Known limits

- **Pieces under tables** or in the corners of the dining room may not be seen from any lookout.
- **One piece per hand.** A trip takes two pieces at most, so the third always needs a second trip from the
  dining room to the corridor.
- **Development data.** The controller was developed in 1.5 hours on the public starts 301-310 only. The
  development rounds and the scored run are in the task's report.
