# a01: plates from the bar and the countertop onto the pantry's shelves

First controller for `putting_dishes_away_after_cleaning` (challenge task 11, scene `house_single_floor`). The code
base is `store_batteries` x02 (map prior, scan-matching localisation, whole-body IK, arm planner over the map, the
hinge skill of `storing_food` for the pantry door). New: `dish_pipeline.py` (the task), `dishes.py` (the plate
skills), `pantry.py` (the door in two stages); `observer.py` holds the plates' sizes and surfaces;
`trash_observed.observed_pipeline` calls `dish_pipeline.full_task`. The battery code of x02 is still in the folder,
unused.

## The task (dev probe 301)

- **Eight plates** (`plate.n.04_1..8`), 0.24-0.31 m across, 2.3-3.4 cm high, four models. Four stand on the bar
  `bar_udatjt_0` (top 1.052 m, along its south side, y about 0), four on the countertop `countertop_kelzer_0` by the
  south wall (top 0.888 m, y -1.7..-2.0). Between them runs the kitchen's aisle (y -1.585..-0.305, middle -0.945).
- **Goal:** one cabinet with all eight plates inside; six cabinets are in the task. Literals already true at the start
  (all cabinets not open) give nothing, so Q = plates inside the best cabinet / 8.
- **The cabinet used:** the pantry `bottom_cabinet_fancyy_0` (`cabinet.n.01_2`) at the aisle's east end. Its east bay
  (x 8.76-9.32, y -2.17..-1.63) has one full-height door (`j_door2`, hinged at its east edge x 9.328, opens to the
  north) and shelves at 0.849, 1.162, 1.479 and 1.792 m.
- **Inside** (oracle jobs `dev/jobs/inside_test.py`, in1/in2): a plate on the east bay's 0.849, 1.162 or 1.479 shelf
  counts inside the pantry, also several in a jumbled pile; its middle must be some 8 cm in from the bay's front
  (y -1.68 counted, -1.62/-1.65/-1.66 did not). The goal options' order (and the literals' order in an option)
  differs between loads of the same instance: read the goal by the best option, not by its index.
- **Horizon:** 16430 steps.

## What it does (`dish_pipeline.full_task`)

1. **Localise** (global, looking around: `trash_observed.localize`).
2. **Survey** from the aisle's middle: facing north along the bar (x 8.3, 7.4, 6.2, 5.0, or the other way round from
   the nearer end), one turn at the end, and back facing south along the countertop. A plate is a patch 1-7 cm over a
   known surface whose top is 1.5-5.5 cm over it and which is 0.10-0.40 m across (`observer.floor_things` with the
   plates' numbers); sightings touching the image border are not used; sightings within 0.12 m are one plate.
3. **The pantry door**, two stages (`pantry.py`):
   - the left hand takes the handle bar from the north and leads it round the hinge line (the hinge skill): the door
     came to 23-54 deg (the leaf's free end, beyond the handle, comes against the forearm that holds it);
   - then the right hand, closed, goes into the gap between the leaf and the bay's front from the west (fingers along
     the leaf towards the hinge, 3.5 cm off its inner face, at 16 deg) and is led round the hinge line to 90 deg,
     pushing the leaf ahead. The leaf is put into the map as boxes (where the first stage left it, then where the
     push left it) so that the arm planner keeps the arms and the plates off it.
   - then the head camera counts points where the open leaf would stand and where the shut leaf would stand; if
     the door looks shut (r6 301: the handle held, the hand went to 68 deg, the door stayed shut), both stages are
     done once more.
4. **Each plate**, nearest first: the base goes to the aisle's middle opposite it, looks again (the plate 0.9-1.1 m
   off and whole in the image: its place relative to the base as it stands now), drives straight in to 0.36 m from
   the surface's edge, pinches the plate's near rim with the fingers straight down (`dishes.rim_grip`/`pinch`: the
   fingers' ends 1.8 cm over the surface, 1 cm in from the plate's edge, closing along the way to its middle), lifts
   8 cm, draws it in level, and backs straight out to the aisle's middle.
5. **Delivery** (`dishes.place`): from a stance west of the open leaf the plate is brought, level, to 0.3 m in front
   of the bay at the shelf's height, slid straight in until the fingers are 4 cm in front of the bay (further in, the
   wrist meets the shelf above), let go 3 cm over the shelf, and the closed hand, fingers 45 deg down, pushes the
   plate's near rim further in. Slots (`SLOTS`): two plates a shelf, one behind the other (the first pushed 0.22 m in,
   to the back; the next let go in front of it and pushed 7 cm): 1.162, 0.849, 1.479, then piles. A plate let go
   over another at the bay's front tipped forwards off the shelf (r3 301, twice).
6. **A missed pinch** (the fingers closed on nothing) is tried again at once from the same stance, the plate's
   middle taken 2 cm nearer, then 2.5 cm further.
7. **The plate's place** for the pinch comes from the aisle look: its near edge (the patch's point nearest the camera,
   well seen) plus its radius along the line of sight. The patch's middle came out 4.5 cm towards the camera on a
   countertop plate, and the pinch missed its rim (r3 301).

## Localisation in this kitchen

- After the survey the base goes to (5.60, -0.945) and localises afresh, globally, looking around (the start fixes
  there were right): the survey's turn at the east end once left the pose 0.16 m off along the aisle for good, and the
  door and the first plate then failed (r5 302).

- In the aisle's east part (x > 7.6) the scans matched the map poorly (score 0.6 against 0.98 elsewhere) and the
  corrections walked the map pose 0.3 m along the aisle after a turn on the spot, and 0.84 m on the way to the pantry
  (r1 301). There the pose is kept by dead reckoning alone (`FREEZE_EAST`); turns on the spot are made slowly
  (0.45 rad/s); the survey turns once, at the west end. With this the pose stayed within 2-4 cm (r2 301).
- Between x 6.4 and 7.6 x is held by dead reckoning (`LOCK_X`), y and the heading are corrected: arriving at the
  door the pose was once 0.17 m off along the aisle after the fresh fix (r7 301).
- Corrections are also off outside the aisle's middle band (y -1.12..-0.78): at a stance 0.36 m from the bar the pose
  went 0.45 m off (r3 301).
- Along the aisle's middle the base turns only west of x 7.4 and slides sideways east of it (`go`): turns in the
  dead-reckoned east part put the pose 8-10 cm off at the pantry, and two plates were let go short of the shelf and
  fell to the floor (r4 301).

## Development record (dev server, public_test, start 301)

- Oracle trials (`dev/jobs/plate_trial.py`): the rim pinch from 0.36 m held a bar plate and a countertop plate (pt1,
  pt2); a plate set down 4 cm in front of the bay from (8.55, -1.10, -75 deg) lay on the 0.849 shelf inside the pantry
  (pt4). A direct joint move to the release pose stalled the plate against the shelf (pt3): hence the slide in.
- Door (`dev/jobs/door_trial.py`): handle lead from (8.583, -1.093, -75 deg) 28 deg (dt1); from (8.516, -1.161,
  -45 deg) 54 deg (dt2, dt6), 23 deg with the grip 1.2 cm lower (dt5), 37 deg in r2; from (8.50, -1.05) the door did
  not follow the hand (dt3, dt4). Push with the left arm (dt6): the arm knocked the leaf from 54 to 14 deg on its way
  in (now the right arm, and the leaf in the map).
- r2 301: survey found the plates (pose within 4 cm), door 37 deg, a bar plate taken, but it could not be brought to
  the bay: the leaf stood in the way and the plate pushed it nearly shut.
- r4 301 (slots, near-edge plate place): door 0.65 rad after the handle lead, 1.585 rad (91 deg) after the push; 3 of
  8 plates inside at the end (two on the 1.162 shelf, one on 1.479); two plates set on the 0.849 shelf fell to the
  floor (the map pose 8-10 cm off at the pantry, after turns in the dead-reckoned east part); one on 1.479 not pushed
  in (no arm path) and not inside; two countertop plates at the west end missed twice each (fingers closed on
  nothing). Episode ended at step 12794 (every known plate tried twice).
- r5 302 (turns only west of 7.4): the survey's turn sequence at the east end left the pose 0.16 m off; the handle
  lead and the push then did not move the door (0.01 rad); stopped. Hence the fresh global fix after the survey.
- r6 301 (with the turn-west-of-7.4 rule and the fresh fix after the survey): the fix after the survey was right to
  3 cm; the handle lead held the handle but the door stayed shut (0.14 rad), the push found nothing to push, the plate
  met the shut door; the pose went 0.2-0.29 m and 20 deg off in the east part again after the sideways slides. Hence:
  the slide rule switched off (`AISLE_SLIDE`), and the door check with a second attempt.
- Last round, frozen code (dev snapshot a01_r8 = this folder; r7 = the same without `LOCK_X`):
  - r8 301: 3 of 8 inside (Q 0.375). The first door attempt failed (lead 20 deg, push found no way), the head-camera
    check saw it shut, the second attempt opened it to 93 deg. 7 plates taken and set down; inside: two on 1.162
    (pose error 0.02 m), one on the 1.162 pile. Not inside: the two set on 0.849 and one on 1.479 after a trip from
    the bar's west part (pose 0.08-0.11 m off at the pantry: let go short of the shelf, fell), and one on 1.479 (left
    hand, no push path, middle too near the front).
  - r7 302: 2 of 8 inside (Q 0.25). Door 0.97 rad after the lead, 1.58 rad after the push; two plates on 1.162 inside;
    a phantom plate north of the bar (no stance); countertop plates at the west end missed (retries included); later
    placements with the pose 0.06-0.11 m off at the pantry did not stay.

## Known limits / next steps

- **The pose at the pantry.** East of x 7.6 the pose is dead-reckoned; after a long trip from the aisle's west part it
  arrives 6-11 cm off and the plate is let go short of the shelf. Fix ideas: measure the bay's front (shelf edges at
  y -1.605) with the head camera at the place stance and correct y before the release; or release only with the
  hand's own camera seeing the shelf edge under the plate.
- **The door's first stage** is erratic (0.14-0.97 rad with the same stance); the head-camera check and a second
  attempt rescue it, at about 1500 steps.
- **Pinch misses** on countertop plates at the west end (x 4.6-5.2, the flat model pkkgzc and the wide xfjmld).
- The 1.479 shelf: the left hand has no push path there; its plates stay at the bay's front and do not count.
