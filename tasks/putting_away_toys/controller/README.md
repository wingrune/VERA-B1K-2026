# y01: putting away toys with the trash controller

First controller for `putting_away_toys` (task 54). It is the `clean_up_broken_glass` controller `g01`
(`tasks/clean_up_broken_glass/experiments/candidate/g01`), itself the `picking_up_trash` controller `t01`, fitted
to toy figures and an open toy box. The design is described in the t01 and g01 READMEs:
- a map prior and scan-matching localisation;
- things found by size on floor the map has empty;
- a top-down pinch from the floor with the trunk folded over the base, one thing per hand;
- staging points for turning.

This file lists what differs. The module and variable names still say "can", "soda" and "trash can": here they
stand for a toy figure and the toy box.

## The task

- **Scene:** `house_single_floor`. The robot starts in the living room.
- **The toys:** eight toy figures (`toy_figure.n.01_1..8`, eight models), 10-17 cm across, 4-9 cm high, 0.225 kg. Four
  lie on the living-room floor and four on the dining-room floor.
- **The toy boxes:** two, both open.
  - `toy_box.n.01_1` (model `qgfbdj`, 0.82 x 0.51 m, rim at 0.40 m) stands somewhere on the living-room floor.
  - `toy_box.n.01_2` stands on the dining table. It is not used.
- **The goal:** every toy inside one of the boxes. Q is the fraction of the eight goal predicates met.
- **Horizon:** about 16950 steps (1.5 x the human mean of 11297).

## What is different from g01

- **Sizes.** Things on empty floor are taken for toys when their top is 1-11.5 cm up. They are pinched at 3 cm
  (`observer.py`, `trash_skills.py`).
- **The toy box is a rectangle.**
  - Its rim (0.36-0.43 m up) is taken when its extents along its own two axes come within 12 cm of 0.82 x 0.51 m.
    Its middle is the middle of those extents.
  - While a hand is over it, it is a box for the validator 0.51 m across (its short side).
  - The base stands with its middle 0.84 m ahead, or else 0.78 or 0.90 m. Then the chassis, which reaches 0.42 m
    ahead, stays clear of the box whichever way the box is turned. An offline check found 22-46 arm poses over the
    box at 0.66-0.84 m.
- **Letting go.** The hand stays pointing down. The toy hangs below it, 8 cm above the rim.
- **Two floors.** Toys are looked for on the living- and the dining-room floor.
- **Lookouts:**
  - for toys: three places in the living room, two in the dining room, and one by the sofa;
  - for the toy box: the three living-room places.
- **Eight toys:** up to six search rounds until all eight are in.
- **Map** of this scene: `priors/house_map.npz`, built by `dev/build_scene_map.py --bounds 3.5 -2.8 17.9 5.0`
  without the floors, ceilings, toys and toy boxes.
