# o01: sorting the cans and the bottles with the trash controller

First controller for `sorting_bottles_cans_and_paper` (task 52). It is the `clean_up_broken_glass` controller `g01`
(`tasks/clean_up_broken_glass/experiments/candidate/g01`), itself the `picking_up_trash` controller `t01`, in the
same scene as t01 (`house_double_floor_lower`, its kitchen). It sorts the two soda cans into an ice bucket and the
two wine bottles into the big bucket. The newspaper and the magazine lie flat on the bar and are left: no skill of
these controllers takes a flat sheet. The design (map prior, scan-matching localisation, things found by size, a pinch
from above with one thing per hand, staging points for turning) is described in the t01 and g01 READMEs; this file
lists what differs.

## The task

- **Buckets:** three, on the kitchen floor, open.
  - Two ice buckets (`vlurir`): rim at 0.40 m, 0.34-0.42 m across.
  - One big bucket (`bdhvnt`): rim at 0.52 m, 0.51 x 0.63 m.
- **Cans** (7.5 cm across, 12.7 cm high): one on the floor, one on the kitchen's bar (top at 0.89 m).
- **Wine bottles** (7 cm across, 22 cm high): one on the floor, one on the bar.
- **The newspaper and the magazine:** flat on the bar.
- **The goal:** three buckets, one with both bottles, one with both cans, one with both sheets, each with nothing
  else.
- **The score.** Each of its six ways of assigning buckets has 16 conditions, 10 of them "not inside", which hold from
  the start and count for nothing.
  - Q is the share of conditions newly met, the best way, unless the whole goal is met (Q 1).
  - Cans and bottles sorted, without the sheets, give at most Q 4/16 = 0.25.
- **Horizon:** about 15400 steps (1.5 x the human mean of 10264).

## What is different from g01

- **Two kinds, in turn** (`KINDS` in `trash_pipeline.py`). The cans go to an ice bucket, then the bottles to the
  big bucket.
  - Each kind is a separate run of the trip loop.
  - Each has its own rim height, box for the validator and stance at the bucket (its middle 0.58 or 0.74 m ahead).
  - Each has its own height from which the hand comes straight down: 0.12 m for a can, 0.26 m for a bottle. From
    0.13 m, the open fingers knocked a bottle over.
- **What things are.**
  - A cluster is a can or a bottle by its top's height above what it stands on (the floor or the bar).
  - It is a bucket by its rim's height (0.36-0.44 or 0.48-0.57 m) and width. The middle of the rim's extents is
    the bucket's middle.
- **The bar.** Things are looked for over it too, at 0.90-1.16 m, and pinched 3 cm above its top. The base keeps 0.40
  m from the bar, so a thing there is 0.55-0.90 m ahead when it is pinched.
- **Map** of this scene: `priors/house_map.npz`, built by
  `dev/build_scene_map.py --bounds -1.0 -3.0 10.0 7.0`. Every task object is left out, the ice buckets included:
  with them in the map, no ice bucket was found where it stood in other starts, and from a start next to one no route
  was found.
- **No restaurant zones;** sightings closer than 0.30 m to a bucket's middle are not things on the floor.
- **Keep-out discs:** 0.35 m around things on the floor (t01: 0.62).
- **Lookouts:** four places in the kitchen.

## Known limits

- **The sheets** are not taken, so no episode can succeed. The best Q is 0.25.
- **A bottle knocked over** stays lying. Lying things are not looked for.
