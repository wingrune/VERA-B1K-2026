# v01: mousetraps from the bathroom cabinet to the floor under the sink

Frozen 2026-10-17 01:12 MSK; code identical to dev snapshot `v01_f` (`/data/aermakov/fax/mouse/snap/v01_f`), dev Q 0.983 on 301-310.

Controller for `setting_mousetraps` (challenge task 5, scene `house_double_floor_upper`, robot R1 Pro), scaffolded
from `store_batteries` `x02` (map prior, scan-matching localisation, small things found by size over a known
surface, a pinch from above at the sighted height, trips with two hands). This file lists what is new.

## The task and its score

- **The mousetraps** (`mousetrap.n.01_1..4`, model `mwfwsv`, 88 g): flat, 10.8 x 5.0 cm, 1.8 cm high (a board
  0.65 cm thick with the spring and the bar on it). At the start all four lie on the top of the bathroom cabinet
  `bottom_cabinet_jhymlr_0` (x 1.83-2.22, y 1.43-2.61, collision top 0.8815 m), anywhere on it (public 301-310).
- **The goal:** all four `ontop` the bathroom floor, and two of them `under` or `nextto` the sink
  (`furniture_sink_ojjqku_0`, a wall-hung vanity at the west wall: x -2.42..-1.905, y 1.62..2.83, its underside
  0.352 m up, free floor under it).
- **The score.** Every goal option has 6 literals (4 `ontop floor` + 2 `under`/`nextto` sink), none true at the
  start. Q = literals met / 6: four traps on the floor are worth 0.667, each trap under the sink 1/6 more.
- **What counts** (oracle job `under301`, traps put on the floor by hand):
  - `under` the sink: the vertical line up from the middle of the trap's box meets the vanity. Middle at
    x -1.900 yes, -1.897 no; along y anywhere from 1.66 to 2.78. Upside down also counts.
  - `nextto` the sink never holds for a trap on the floor: next-to compares the boxes' gap (here the 0.33 m from
    the trap up to the vanity's underside) with a sixth of their mean size (0.15 m).
  - `ontop floor` fails for a trap that rests tilted on another one: the traps are laid apart.
- **Horizon:** 15294 steps.

## What it does

1. **Localise** globally (turning on the spot; the bathroom's scans are unambiguous: right to 1-2 cm), fold the arms,
   look around once.
2. **Find the traps** (`observer.py`): the cabinet top is the only surface (`SURFACES`). The camera sees the top at
   0.8884 m, 7 mm over its collision top: the traps' boards sink into it and only the spring and the bar show,
   0.5-1.4 cm over it. A trap is a patch of points 0.45-6 cm over the seen top, its top 0.7-3 cm over it, 3.5-14.5 cm
   across; a wider patch (two traps touching, 10 cm apart on some starts) is split by 2-means along its long axis.
   The patch's principal axis is the trap's long axis (modulo half a turn). Offline on saved views of 301, 302 and
   304 (`dev` job `views*`, true base pose) all four were found in every view, to 1 cm and 3 degrees. The cover check
   of the batteries' controller is off (the wall behind the cabinet was taken for a cover, 304). Lookouts west of the
   cabinet at (0.85, 2.05), (0.85, 2.35), (1.05, 2.2).
3. **Pinch** (`trash_skills.plan_trap_pinch`): from above, the fingers closing across the long axis, their centre 2 mm
   over the collision top (the board is 6.5 mm thick), **3 cm from the trap's middle** along its axis; the middle
   then lies 3 cm out along the tool's x axis. The other traps are 8 cm boxes for the validator.
4. **Lay under the sink** (`trash_skills.plan_lay` / `lay_trap`, `trash_pipeline.deliver`): one trap per stance.
   The hand comes down 14 cm in front of the place, 5.5 cm over the floor, and slides in level to 2.5 cm over the
   floor, the trap's axis pointing in under the vanity; it opens, slides back out, folds. The trap's middle is
   aimed at x -1.945 (4 cm in under the vanity's front), places 0.15 m apart along it.
   - Straight in front of the vanity there is no plan: reaching for the floor the trunk bends over the target and
     hits the vanity's top (offline, bases 0.45-1.0 m from it).
   - From the side there is: the base parallel to the vanity's front, 0.45-0.52 m from the place across, the left
     hand facing south (places y 1.80-2.25), the right hand facing north (places y 2.40-2.70) (`LAY_SPOTS`,
     `LAY_ARM_AT`). Oracle job `trial301`: two traps pinched off the cabinet top, carried folded and laid this way
     ended at (-1.952, 2.087) and (-1.945, 2.234), flat, under the sink and on the floor.
   - A trap that cannot be laid under the sink goes on the open floor 0.15 m in front of the bathtub's side
     (`FLOOR_SPOTS`): it still counts as on the floor.
   - Last resort (`trash_skills.drop_on_floor`): a trap that could not be laid anywhere is held 0.25 m over free
     floor (map) in front of the base and let fall (306: the right hand stalled three times on its way down to the
     open floor, and the trap stayed in the hand to the end).
5. **Order and retries** (`trash_pipeline.full_task`): two traps per trip (left and right hand), the tour planner
   of x02. A trap one hand has missed twice is left to the other hand; four misses leave it alone. After the four
   are out (or left alone) two more rounds of lookouts (`SEARCHES`), then the episode ends.
6. **Tracks** (`trash_observed.py`): a known trap that should show in the head image but does not counts a miss
   only from 0.9-1.3 m (`EXPECTED`; from 1.5-1.6 m a trap shows with 11-13 points or not at all), and is
   forgotten after 10 such misses (307: tracks were dropped from the staging points 1.6 m off and re-found under
   new numbers, and fetches gave up on them).

## Known limits

- **A trap deep in the cabinet's north-east corner** has no stance: 304's trap at (2.166, 2.507), between the
  cabinet and the shower door (offline stance search: no room at any stance from which the arm reaches). It stays
  on the cabinet: 304 reaches 5/6 (the other three are laid under the sink, so both `under` literals hold). 305's
  trap at (2.173, 1.483) by the south wall had no stance in rounds d (no trap), and was taken in round f.
- **Only one place under the sink for the right hand** (y 2.70): from the others the base would stand by the
  serving cart. A second trap in the right hand goes to the open floor (counts as on the floor only).

## Development (dev server, whole runs `dev/jobs/chain_observed.py`, public starts, Descartes)

Q = literals met / 6 from the `RESULT` line (best goal option). Final code = snapshot `v01_f` (round f).

| start | round f (v01_f, frozen) | earlier rounds |
|---|---|---|
| 301 | **1.0** (3 under, 1 on the open floor) | b 0.667 (one trap left: pinch 5 mm too low) |
| 302 | **1.0** | b 0.833 (four misses of one trap, all by the right hand) |
| 303 | **1.0** | c 1.0 |
| 304 | **0.833** (north-east corner trap left on the cabinet) | d 0.833 |
| 305 | **1.0** | d 0.833 (south-east corner trap: no stance) |
| 306 | **1.0** | d 0.833 (last trap stayed in the hand: floor lay stalled; now the drop fallback) |
| 307 | **1.0** | e 1.0; d: tracks dropped from 1.6 m, fetches gave up (stopped) |
| 308 | **1.0** | e 1.0 |
| 309 | **1.0** | e 1.0 |
| 310 | **1.0** | |

**Mean over 301-310: 0.983 (9 of 10 successes).** Rounds: b = v01_b (first whole runs), c = + lay shifts and
"other hand after two misses", d = + grip 6 mm higher, fewer search rounds, e = + track misses only from 0.9-1.3 m,
f = + drop-on-floor last resort. The last resort was exercised in the oracle job `a_drop301` (lay places switched
off: both traps were let fall in front of the base and lay flat on the floor, `ontop` true); in round f no start
needed it.

Dev jobs in `dev/jobs`: `under_study.py` (what counts as under; run as `under301`), `cabinet_views.py` (saved head
views for the detector; `views301/302/304`), `trap_trial.py` (pinch + lay with oracle poses; `trial301`, and with a
height per case `trial302`), `drop_fallback_trial.py` (the last resort through `deliver()`; `a_drop301`). Map: `dev/build_scene_map.py --bounds -2.7 1.0 2.5 4.4
--open-furniture furniture_sink_ojjqku_0` with `mousetrap` and `downlight` skipped.
