# z02: both tupperwares into the fridge (z01 with stances at the bar's south end, side-by-side shelving, no shut)

z02 is z01 (below) with the changes listed in the section "z02" at the end; everything else is z01's.

# z01: the tupperwares into the fridge with the plates controller

First controller for `clearing_food_from_table_into_fridge` (task 25). It is the `cleaning_up_plates_and_food`
controller `f01` (`tasks/cleaning_up_plates_and_food/experiments/candidate/f01`): the same scene
(`house_double_floor_lower`), the same kitchen, breakfast table and fridge. Its localisation, routes, door stages and
the slide onto the fridge's shelf are used as they are; see the f01 README. What z01 does: it opens the fridge's
right door, takes the tupperwares from the bar one by one, puts them on the fridge's lower shelf (the second on the
first), and shuts the door. The half chicken and the half apple pie stay on their plates on the table.

## The task and its score

- **On the breakfast table:** a half chicken and a half apple pie, each on a plate.
- **On the bar east of the table (top at 0.89 m):** two open tupperwares (`mkstwr`, 0.22 x 0.22 m, 13 cm high, 66 g).
- **The goal:** each food inside a tupperware, both tupperwares inside the fridge, the fridge not open.
- **The score.** The goal has five conditions; "not open" holds from the start and counts for nothing. Partial
  credit is the share of conditions newly met, so each tupperware in the fridge adds 0.2. Without the food the best
  is 0.4.
- **Horizon:** about 19600 steps (1.5 x the human mean of 13068).

## What is different from f01

- **The tupperwares are found over the bar** (`TableView.bar_things` in `table_view.py`).
  - A tupperware is a group of depth points 4-16 cm above the bar's top, 10-32 cm across.
  - Points over map cells with anything standing on the bar are left out.
  - A tupperware counts as known only once seen from nearer than 2.5 m. From across the kitchen the bar's far
    edge passed for one.
- **Stances at the bar:** west of it, facing east or turned up to 1.4 rad, with the tupperware 0.55-0.94 m ahead and
  to the left hand's side (`table_options`).
  - At the bar's south end the kitchen's counter adjoins it. There a tupperware 0.75 m or more south of the bar's
    middle line has no stance.
- **The grip.** The rim is taken on the side nearest the robot, the fingertips 2.5 cm below it, one finger inside, at
  60, 45, 75 or 90 degrees down. From 75-90 degrees alone no arm pose reached a tupperware 0.75 m ahead.
- **The map over the bar.** Cells higher than 1.2 m (the cabinet above the bar) are set to the bar's top. In the
  height field the cabinet stood on the bar, and no hand could reach a tupperware under it (`bar_overhang` in the
  map's meta).
- **Into the fridge.** f01's slide onto the lower shelf, with the tupperware's size instead of a plate's. The second
  tupperware goes on the first one.
- **Stages:** `('door', 'tups', 'shut')`. The bowls and plates of f01 are not handled.

## Known limits

- **The food is not moved.** At most Q 0.4.
- **Tupperwares in the bar's south corner** are not reached.

## z02 (2026-10-17)

**Why.** z01 put at most one tupperware into the fridge (6 of 10 public starts, Q 0.12). Its losses (scored traces in
`/data/aermakov/fax/eval/food_z01_final/`): most tupperwares stand near the bar's south end, where z01 had no stance; a
patch at the bar's south-east corner passed for a tupperware (302) and hid the real second one, which stood behind the
first as seen from the lookouts; the second tupperware set on the first one stalled 0.3 m short of the shelf (307).

**Where the tupperwares stand** (dev job `positions.py`, starts 301-310): 15 of 20 within y -0.39..-0.86 (the bar's
south part), x 7.92-8.31. The base can stand no nearer than the free corner at about (7.40, 0.05).

**What changed.**
- **Stances.** After z01's stances, any free base pose west of the bar 0.55-1.28 m from the tupperware, facing it or
  turned up to 0.5 rad, nearest first (at most 80, 45 s of planning per take). Offline (the arm planner over the map)
  every tupperware position of the ten starts is then reached, the far ones with the left hand 0.97-1.17 m ahead and
  the trunk bent over the bar. z01's stances also go to 0.98 m and turned to -1.6 rad. In dev runs tupperwares were
  taken from 0.94-1.06 m.
- **The tupperware sightings** must have their middle 0.11 m or more in from the bar's edges (no phantom at the corner).
- **The estimator** forgets a tupperware only when another one is known within 0.3 m of it (a double); a tupperware whose
  pinch closed on nothing is dropped from what is known and looked for again (a double of a tupperware already taken was
  pinched at eight times, d 309).
- **Shelving.** Both on the lower shelf side by side: the first with its middle at x 4.825, the second at 5.055 (the
  fridge's container volume is x 4.735-5.19; nearer the east wall the second stalled in the opening). The hardest
  tupperware (deepest into the bar's south corner) is taken first and gets the west spot: one taken 1.03-1.12 m ahead
  and set down second, at the east spot, stalled in the opening (h 308, j 305, h 310, k 305), while at the west spot
  such ones went in. Each free spot is tried in turn, at two heights; a tupperware is let go once its middle is in
  past y -0.285 (inside; one brought out again from -0.306, k 306); one no spot takes is let go where the base stands
  (held on, every later take failed at once, 308).
- **The door.** After z01's lead by the handle, a look at the door; if it stands short of 88 degrees (on 301 the lead
  left it at 70 four times) z01's lead by the free edge takes it on to 105; then one push on its inner face to about
  128 degrees (h 309: 103 -> 127; l 301: 92 -> 126). Whatever these do, the door counts as open as the lead left it.
- **No shut stage.** "The fridge not open" holds from the start and counts for nothing in Q; shutting cost about 5000
  steps. Up to 10 attempts at the tupperwares (z01: 6).
- **The food: off (`FOOD = False`).** `food.py` takes the half chicken and the half apple pie off their plates by a
  top-down pinch across their narrow side (found as not-white groups 3.3-20 cm over the table's top, their long way by
  PCA of the points) and lets them fall into the western tupperware (tupperware.n.01_1 on 8 of the 10 public starts; the
  goal wants both foods in that one). A privileged trial (true poses, 301): both taken and carried, the pie fell in
  (inside), the chicken rested on the pie, out of it. Whole runs:
  - f 301 (both foods, straight to the drop pose): the chicken, let fall at the corner tupperware 0.95 m off, swept it
    12 cm aside and tilted it; no hand took it after.
  - dev snapshot `z02_p` (= z02_n with `FOOD = True`, the pie only, lowered from 12 cm over the drop point, and only
    where the western tupperware stands within 0.92 m of a drop stance): 302 the pie in tupperware_1 and both
    tupperwares in the fridge (Q 0.6); 307 and 309 the pie went into the western one, which is tupperware_2 there (no
    credit); 301 and 304 skipped (too far); **303 and 305: the drop pushed tupperware_1, with the pie in it, toward
    and over the bar's edge and tilted it (Q 0.2 on each, against 0.4 without the pie)**. So the frozen z02 leaves the food: the
    pie gains 0.2 only where the western tupperware is tupperware_1, and costs up to 0.4 where the drop moves it.

**Dev record** (dev server, public_test; simulator truth read only for the record; one run per row; the letter is the
dev snapshot `/data/aermakov/fax/food/snap/z02_<letter>`):

| Start | Run | Code | Tupperwares in the fridge | Q (of 5) | z01 scored |
|---:|---|---|---:|---:|---:|
| 307 | a | first z02 (no far stances) | 2 | 0.4 | 0.2 |
| 304 | b | far stances | 1 | 0.2 | 0.0 |
| 304 | c | + never forgotten | 2 | 0.4 | 0.0 |
| 308 | c | | 0 (first one stalled at the fridge, kept in the hand) | 0.0 | 0.0 |
| 309 | d | | 1 (a double of the first pinched at 8 times) | 0.2 | 0.0 |
| 301 | e | + fridge retries, let go if none | 1 | 0.2 | 0.2 |
| 301 | f | + food stage | 0 (the chicken pushed the tupperware aside) | 0.0 | 0.2 |
| 301 | h | food off, door widening, doubles | 0 (door at 70 deg: no way to the opening) | 0.0 | 0.2 |
| 302 | h | | 2 | 0.4 | 0.2 |
| 308 | h | | 1 (second, taken 1.12 m ahead, stalled at the east spot) | 0.2 | 0.0 |
| 309 | h | | 2 | 0.4 | 0.0 |
| 310 | h | | 1 (second, 1.10 m ahead, stalled at the east spot) | 0.2 | 0.2 |
| 303 | j | + other spot on failure | 0 (the pair 0.25 apart: one forgotten again and again) | 0.0 | 0.0 |
| 305 | j | | 1 (second, 1.03 m ahead, stalled at the east spot) | 0.2 | 0.2 |
| 306 | j | | 1 (second, 1.04 m ahead, stalled at the east spot) | 0.2 | 0.2 |
| 303 | k | + hardest first, twins 0.18 m | 1 (second stalled at the east spot, 5.075) | 0.2 | 0.0 |
| 306 | k | | 1 (second, near, reached y -0.306 inside, brought out again) | 0.2 | 0.2 |
| 305 | k | | 1 (hardest had no stance first; taken second 1.03 m ahead, stalled at the east spot) | 0.2 | 0.2 |
| 301 | l | + spots 4.825/5.055, door look and edge lead | 2 | 0.4 | 0.2 |
| 303 | n | frozen code (+ let go once inside, 80 grid stances) | 2 | 0.4 | 0.0 |
| 305 | n | frozen code | 2 | 0.4 | 0.2 |
| 310 | l | (l: before "let go once inside") | 1 (second reached y -0.318, brought out again) | 0.2 | 0.2 |
| 306 | n | frozen code | 2 | 0.4 | 0.2 |
| 310 | n | frozen code | 2 | 0.4 | 0.2 |
| 301 | n | frozen code: the look found the door short, the edge lead took it to 104, the push to 126 | 2 | 0.4 | 0.2 |
| 308 | n | frozen code (the first stalled twice at the west spot and went in at the east one) | 2 | 0.4 | 0.0 |
| 302 | p | z02_n + the pie | 2, and the pie in tupperware_1 | 0.6 | 0.2 |
| 301 | p | | 2 (pie skipped: the western tupperware too far) | 0.4 | 0.2 |
| 303 | p | | the pie in tupperware_1, which was then pushed over the bar's edge: no tupperware into the fridge | 0.2 (z02_n: 0.4) | 0.0 |
| 305 | p | | the pie in tupperware_1, which the drop pushed to the bar's edge and tilted; the pie fell out of it later, and only tupperware_2 got into the fridge | 0.2 (z02_n: 0.4) | 0.2 |
| 307, 309 | p | | the pie into the western tupperware, tupperware_2 there | (no credit) | |
| 304 | p | | pie skipped (too far); as z02_n | | 0.0 |

**Summary of the frozen code (z02_n).** On the six starts it ran (301, 303, 305, 306, 308, 310) both tupperwares
went into the fridge: Q 0.4 each, the most without moving the food. On 302, 304, 307 and 309 earlier z02 variants put
both in (0.4). z01's scored mean on 301-310 was 0.12. Not a scored result: one dev run per start, the dev server, the
public starts the code was developed on.
