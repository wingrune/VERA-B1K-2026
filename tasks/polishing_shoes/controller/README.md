# h02: the shoes next to each other and to the footstool, with the glass controllers

First controller for `polishing_shoes` (task 79). It is the `dispose_of_glass` controller `w01`
(`tasks/dispose_of_glass/experiments/candidate/w01`), in the same scene (`hotel_suite_large`), here in its
bedroom. It finds the two shoes on the floor and takes each by its collar. Then it sets them down side by side on
the floor next to the second footstool. The dust is not brushed off. The design (map prior, scan-matching
localisation, things found by size, a pinch by the rim nearest the robot with one finger inside, staging points for
turning) is described in the w01, g01 and t01 READMEs; this file lists what differs.

## The task and its score

- **The shoes:** two (`walker`, about 0.2 m long, collars 9 cm up, 0.5 kg), dusty, on the bedroom floor.
- **The scrub brush** stands on a nightstand.
- **The footstools:** two ottomans (9 kg), at the foot of the beds.
- **The goal's conditions:**
  - the brush on the stand;
  - each shoe not covered with dust;
  - the shoes next to each other;
  - one footstool next to both shoes.
- **The score.** The brush is on the stand from the start, so that condition counts for nothing. Partial credit is
  the share of the six newly met:
  - each "next to" adds 1/6;
  - the dust is the other two sixths.

  h02 reaches at most 0.5.
- **Horizon:** about 16110 steps (1.5 x the human mean of 10738).

## What is different from w01

- **The shoes.**
  - A shoe is found as a thing on empty floor whose top is 7.5-10.5 cm up.
  - It is pinched by its collar, 3.5 cm from the middle of its opening on the side nearest the robot (one finger
    inside), with the fingertips 6.5 cm up.
  - The other shoe is a box for the validator meanwhile.
- **Where they go.**
  - A fixed place on the floor east of the second footstool (`ottoman_hnmndv_1`), by its middle (`SHOE_SPOT`).
  - Each hand sets its shoe down 0.10 m to its own side, the sole 3 cm above the floor (`SIDE_BY_SIDE`).
  - "Next to" counted only for gaps of about 5 cm or less: with the place 6 cm further east and the shoes 0.26 m
    apart, one shoe and the pair did not count (start 303 in development).
- **Map** of the scene, built without the shoes and the brush (`dev/build_scene_map.py --bounds -3.8 -5.8 2.8 6.3`).
- **Lookouts** in the bedroom.

## Known limits

- **The dust is not brushed off,** so no episode can succeed.
- **The place is fixed:** it was measured on start 301. A start whose footstools stand elsewhere loses the
  footstool conditions.
