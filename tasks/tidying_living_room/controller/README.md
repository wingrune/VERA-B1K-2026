# v02: v01 plus the hardback into a bookcase

Second controller for `tidying_living_room` (task 53): v01 (below, unchanged: the pot plant onto the coffee table)
with a phase before it that puts the hardback into a bookcase (`flat.py`). Q at most 0.5 (plant + hardback).

## The hardback (new, `flat.py`)

- **Why it could not be taken:** it lies flat (0.30 x 0.25 x 0.047 m), wider than the open fingers (10 cm). The
  simulator's assisted grasp (AG) holds a thing when both fingers touch it and the ray between the fingers' grasp
  points (0.4-3.4 cm behind the fingertips) crosses it (`robot._calculate_in_hand_object`).
- **What it does (`flat.hardback_task`, before v01's plant phase):**
  1. looks at the coffee table from north of it (else from its east): head-camera points 2.5-6.5 cm over the table's
     top as that view has it (the top came out 2 cm low from 1-1.5 m), grouped in 2.5 cm cells, the largest patch
     15-40 cm across is the book;
  2. pulls it over the table's east end or north side, whichever its near face is within 16 cm of, and only if the
     edge the hand will hold is under 27.5 cm (the bookcase's compartment is 30.6 cm wide; held by its 30 cm edge it
     met the compartment's side and fell, runs g on 302/303). On 301-310 that is 301 only;
  3. the base comes to the pull stance from a point 0.30 m to the side, looks again there: the book's far face and
     the table's position (from its far end plus its length; the table is not fixed and the base nudged it 5-12 cm
     on the way in, runs e/h/v1 on 301) give the pull's length. The closed hand comes down 6 cm beyond the far face
     and is drawn toward the edge (the book as a box keeps the arm's way over it off it); seen again with the arm
     folded, a second stroke if it came short; aimed at 7.5 cm over the edge;
  4. the hand goes straight up, folds, the base backs 0.25 m, stands 0.60 m out and 0.36 m to the side of the pinch
     point; the left hand, level, fingers closing up and down, comes to the overhang (fingertips 1 cm out from the
     edge; only the part on the table is a box for the arm's way there): AG holds the book;
  5. up 5 cm, back 0.45 m, the book carried 0.80 m high 0.40 m ahead of the base; at `bookcase_otwukr_2`'s east
     compartment (x -3.112) the base faces north, the hand comes down to the middle shelf's height (0.394 + 1.5 cm),
     the base drives 0.30 m forward, the hand opens.
- **Oracle trials** (`dev/jobs/pull_trial.py`, `edge_trial.py`, start 301): the slanted pinch of an edge (`edge_trial`)
  never made AG hold the notebook (one finger on the table); pulled over the edge by 7 cm the hardback was held by
  AG and carried into the bookcase (`Inside` true).
- **Not taken:** the notebook (12 cm, 1.5 cm thick: slid 2-4 cm per pull, pushed back by the approaching hand, AG
  never held it) and the newspaper (not tried).

## Development runs of v02 (whole runs, onboard observation, public starts; truth only logged)

| run | start | hardback | pot plant | literals |
|---|---|---|---|---|
| a | 301 | not seen (1 cm cells broke the patch) | on the table | 1/4 |
| b, c, d, e | 301 | the stroke pushed the book west: the table had been nudged; d drove through the table | on the table (b-e) | 1/4 |
| f, g, h | 301-303 | pulls over the edge worked (overhang 4-8 cm); pinches missed at 4 cm or the arm knocked the book off; on 302/303 AG held it but it fell at the bookcase (held by its 30 cm edge) | on the table | 1/4 (301 h: 0) |
| v1, v2 | 301 | pulled 10-12 cm over the (nudged) table's edge; no plan for the pinch (the whole book was a box) | | |
| **v3 (frozen)** | **301** | **pulled 18 cm, held by AG, carried, inside `bookcase_otwukr_2` (leaning 58 degrees)** | **on the table** | **2/4 (Q 0.50)** |
| v3 | 302, 305 | gives up at once (held edge would be 30 cm / far from the edges) | on the table | 1/4 each |
| v3 | 303 | pulled north, held, carried, not inside (its held edge was seen under 27.5 cm, is 29.7 cm) | on the table | 1/4 |
| v3 | 310 | pinch missed | on the table | 1/4 |

Dev mean over the five v3 starts: (2+1+1+1+1)/4/5 = Q 0.30; on the ten public starts v02 adds the hardback only
where the held edge is short (301), so about Q 0.2-0.225 expected there (v01 scored 0.20).

# v01: the pot plant onto the coffee table with the trash controller

First controller for `tidying_living_room` (task 53). It is the `clean_up_broken_glass` controller `g01`
(`tasks/clean_up_broken_glass/experiments/candidate/g01`), itself the `picking_up_trash` controller `t01`, in scene
`house_double_floor_upper`. It finds the pot plant on the floor, takes it by its leaves from above and lets it go
over the coffee table. The other three things are left (see Known limits). The design (map prior, scan-matching
localisation, things found by size on empty floor, a pinch from above, staging points for turning) is described in
the t01 and g01 READMEs; this file lists what differs.

## The task

- **Goal:** four conditions, each worth 0.25:
  - the pot plant on the coffee table;
  - the newspaper on the coffee table;
  - the hardback inside a bookcase;
  - the notebook on the desk.
- **The pot plant** (`jatssq`, 0.30 m across, 0.29 m high, 1.5 kg) stands on the floor north of the living room.
- **The newspaper** lies flat on the floor.
- **The hardback** (0.30 x 0.25 x 0.04 m) and **the notebook** (0.12 x 0.12 x 0.01 m) lie flat on the coffee table.
- **The coffee table** (0.36 m high) stands at the same place in every start seen: x -4.41..-2.96, y -2.82..-2.02.
- **Horizon:** about 18850 steps (1.5 x the human mean of 12566).

## What is different from g01

- **The pot plant.**
  - It is found as a thing on empty floor whose top is 0.27-0.31 m up.
  - It is taken with the fingertips 6 cm below the leaves' top, the hand coming down the last 14 cm. The simulator's
    assisted grasp holds the whole plant once both fingers touch its leaves (they stay almost open).
- **The coffee table is fixed:** the plant is let go 6 cm above its top, 0.23 m in from its north edge (`TABLE_DROP`).
  - The base stands 0.66-0.76 m north of that point, facing south.
  - The table is a box for the validator.
  - The hand points down and the plant hangs below it.
- **Map** of this scene: `priors/house_map.npz`, built by
  `dev/build_scene_map.py --bounds -6.6 -5.1 2.6 1.5` without the floors, ceilings and the four things.
- **Lookouts:** the room north of the living room, where the robot starts, and the living room's north side.

## Known limits

- **The other three things are not taken.**
  - The newspaper lies flat and the hardback lies flat on the table, wider than the open fingers (10 cm).
  - The notebook is 12 cm wide and 1 cm thick.

  So Q is at most 0.25.
