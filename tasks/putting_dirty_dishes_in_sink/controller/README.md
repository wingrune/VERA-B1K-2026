# k02: putting dirty dishes in the sink, through the kitchen door safely

k02 is [k01](../k01/README.md) with the kitchen door made safe and the pose fixed where it matters. k01 put two dishes
in on most starts and never more. In development, the robot tipped over or the physics failed at the kitchen door. The
lead's first guess, a pose estimate gone wrong after the doorway, turned out to be a robot lying on its side.

## What k02 changes

All of it is in `route()` and the functions next to it in `trash_pipeline.py`, the estimator in `trash_observed.py`
and the localiser in `localizer.py`.

- **Why k01 failed at the door (dev runs with the door's angle and the robot's tilt logged).**
  - Going west, the chassis pushes the leaf open. It swings on to 85-90 degrees (its stop) and stays there, a fin
    0.8 m long sticking out west into the kitchen from the hinge.
  - k01 then turned 0.4 m from the fin's free end and drove up the kitchen's corridor past it. Coming back, it came
    down past it again. The chassis rode up on the leaf, which is 3 cm above the floor, and tipped over: 80-90
    degrees on dev 305 and 307. On scored 302 and 307 the physics failed instead.
  - Pushing the leaf also made the wheels slip. The pose estimate was 0.14-0.26 m off after the way west.
- **The way west** (`go_west`):
  - Backs up to the vestibule's east end (x -4.25), facing west. It takes its lateral position and heading from the
    vestibule's two walls, seen by the head camera 0.55-1.0 m ahead (`doorway_fix` in `trash_observed.py`). It
    lines up, then takes a second look.
    - On dev runs the estimate was within 1 cm and 0.2 degrees of the truth afterwards.
    - The map-based estimate was 2-4 degrees and up to 7 cm off there. The chassis has 3.5-4.5 cm to the doorway's
      posts.
    - On three dev runs the chassis then hit the north post, or met the leaf at its hinge with the north corner. The
      base stuck there and its estimate drove on.
  - Drives through on y -3.605, nose first, with the map-based corrections paused. The chassis's north corner pushes
    the leaf 4.5 cm off its hinge.
  - Ends at x -6.90, its rear 0.25 m past the fin's free end, and stands 45 steps while the localiser settles. The
    leaf turned the base up to 11 degrees on the way.
  - Then slides, still facing west, straight south to y -4.30 and east to (-6.65, -4.30), 1.15 m from the fin's free
    end. It settles again there, and only then turns.
  - Turning at the end of the way west, the arm caught the leaf's edge and the base tipped 43 degrees (dev 305).
    Sliding off diagonally at once, the arm dragged along the leaf and the base tilted 13 degrees (dev 307).
  - Every route planned on the kitchen side keeps the base centre 0.48 m off the fin (`LEAF`).
  - **Held up on the way** (`watched`).
    - If the head camera's depth image stops changing while the base is driven, the base is held up. While moving,
      the image changes by 0.03-0.29 m per second; held up, by 0.0001.
    - The estimate is put back to x -5.10, where the base stands when held up at the doorway. The base backs up to the
      vestibule, takes its pose from the walls again, and tries once more.
    - Tested on dev 302 with the first way west deliberately 8 cm north. Held up at the post, the base was detected,
      backed up, re-fixed (5.8 degrees and 7 cm) and passed, its estimate 1.6 cm off at the end.
- **The way east** (`go_east`): comes down the kitchen's corridor to (-6.80, -4.45), south of anything the leaf can
  reach, facing north.
  - Slides east to x -5.95. Drives north until its front is at y -3.24, which pushes the leaf to its stop whatever its
    angle.
  - Backs up to y -3.515 and slides east through the doorway, still facing north. That way the chassis is 0.64 m
    long, which leaves 4.5 cm to the south post and to the leaf.
  - Stands 45 steps in the dining room while the localiser settles: sliding sideways, the dead reckoning fell 0.1 m
    short.
- **A fresh global fix in the dining room** (`relocalize`), as at the start: three views a third of a turn apart.
  It is taken before each way west, at the end of each way east, and before each fetch.
  - Near the booth the tracked pose drifted 0.5-1.8 m and 10-14 degrees. The causes were bumps into the booth's table
    and chairs, and an arm that did not fold in front of the camera.
  - The fix is taken when it is unambiguous and within 2.5 m and 30 degrees of the tracked pose. A clear fix (score
    over 0.85, margin over 0.20) is taken however far off it is. On dev 304 the tracked pose was 1.6 m and 76 degrees
    off after a bump into the booth, and the fix was right to 1 cm.
  - It is skipped where the map shows less than 0.40 m of room, chairs included. A turn beside the booth, 0.04 m from
    it, tipped the robot over (dev 304).
- **The localiser leaves out what it sees where the leaf swings**, west of the frame's middle, in every height band.
  The leaf is 2 m tall and not in the map. With only the lowest band filtered, the closed leaf and the moving leaf
  threw the estimate 0.2 m and 13 degrees off at the doorway.
- **Drops aimed at the basin's middle**, y 0.85 instead of 0.80. A plate let go at 0.80 ended on the basin's front
  edge. The base stands as before: the basin is 0.76-0.80 m ahead (`DROP_ALSO`).
- **A route with no way out** (`planned_route`).
  - It tries once more without the dishes' keep-out discs, which stand on the table out of the chassis's reach.
  - If there is still none, it takes a short straight step (0.25-0.70 m, heading kept) to where the map shows more
    room, then tries again (`escape_point`).
  - On scored 304 and dev 303 the robot found no way from beside the booth with a dish in the hand. It stood still
    for the rest of the run.

## Development runs (2026-10-16, not scores)

Each run is one whole run per start in a dev server (`dev/jobs/chain_observed.py`), from onboard observations. The
simulator's truth is only logged.

**Snapshots.** k02 is snapshot `k02_l`; its code files are identical.
- `k02_m` is `k02_l` plus a branch that never triggered in that run; the branch was then removed.
- `k02_k` lacks only the new ending of the way west: it ended at x -6.80 and slid off diagonally at once.
- `k02_j` also lacks the global fixes before each fetch and after the way east, the acceptance of clear far fixes,
  and the 0.70 m escape step.

| Start | Snapshot | Dishes in | Q | Notes |
|---|---|---:|---:|---|
| 301 | k02_j | - | - | Stopped early, at step 3670, to free the server; no complete run |
| 302 | k02_j | 4 | 1.00 | Done at step 9793; three clean door passes |
| 303 | k02_k | 2 | 0.50 | Script ended at 14151; the other two were left after failed pinches |
| 304 | k02_k | 2 | 0.50 | Tracked pose lost near the booth before the second way west; tipped over |
| 305 | k02_j | 4 | 1.00 | Done at step 11185 |
| 306 | k02_j | 2 | 0.50 | Tipped over on the booth (pose 0.39 m off) on the second trip; run stopped at 12353 |
| 307 | k02_l | 3 | 0.75 | Horizon reached; door passes clean (pose 0.5 cm off after the way west) |
| 308 | k02_k | 1 | 0.25 | Script ended at 11405 |
| 309 | k02_k | 0 | 0.00 | The walls gave no fix; held up at the leaf's hinge four times, each detected and backed out of |
| 309 | k02_m | 2 | 0.50 | Door passes clean (pose 0.7 cm off after the way west); script ended at 15755 |
| 310 | k02_k | 2 | 0.50 | Script ended at 16900 |

**What this shows.**
- **The door.** No run with k02_k, k02_l or k02_m tipped over at the door. On k_309 the walls gave no fix, and the base
  was held up at the leaf's hinge four times; each time this was detected and it backed out. k01 tipped over there on dev 302 and 307 (today)
  and 305 (2026-10-15).
  - With the walls' fix, every way west started within 1 cm and 0.2 degrees of the line.
  - The forced test of the held-up recovery (dev 302, the first way west 8 cm north) passed on the second try.
- **What still loses dishes.**
  - Bumps into the booth during the approach to a dish, after which the tracked pose is off. The fresh global fixes
    catch most of it.
  - Pinches that fail on dishes deep on the table, and one plate thrown onto the kitchen floor by a drop.


## Inherited from k01

First controller for `putting_dirty_dishes_in_sink` (task 85). It is the `dispose_of_glass` controller `w01`
(`tasks/dispose_of_glass/experiments/candidate/w01`), itself from the glass and trash controllers `g01` and `t01`,
fitted to dishes on a restaurant booth and a kitchen sink behind a door. The design is described in the t01, g01
and w01 READMEs:
- a map prior and scan-matching localisation;
- things found by size over a known surface;
- a pinch by the rim nearest the robot, the hand coming straight down;
- one thing per hand, and staging points for turning.

This file lists what differs. The module and variable names still say "can", "soda" and "trash can": here they
stand for a dish and the sink.

### The task

- **Scene:** `restaurant_diner`. The robot starts somewhere in the dining room.
- **The dishes:** two bowls (12-13 cm across, rims at 0.81 m) and two plates (17-24 cm across, rims at 0.78 m).
  They stand on the table of the booth `booth_xzrpar_2` at the dining room's south wall (top at 0.75 m), 0.03-0.55 m
  behind its north edge.
- **The sink:** the kitchen's commercial sink (`commercial_kitchen_sink_xecfyh_0`), about 15 m away.
  - Its basin is 0.43 x 0.37 m inside, at x -11.21..-10.78 and y 0.70..1.07; its deck is at 0.87 m.
  - The faucet stands behind the basin.
- **The kitchen door.** The kitchen is behind a door (`door_mkcndz_0`) that swings into it, from a hinge at its north
  post (-5.43, -3.21).
- **The goal:** every bowl and plate inside the sink. Q is the fraction of the four goal predicates met.
- **Horizon:** about 18220 steps (1.5 x the human mean of 12149).

### What is different from w01

- **Where dishes are looked for.** Over the booth's table only, at 0.762-0.86 m, where a point is 1 cm or more above
  what the map has. A dish is a cluster whose top is 0.75-0.85 m up.
- **The pinch.**
  - **Grip:** fingertips at 0.775 m, 8 cm from the dish's middle on the side nearest the robot. Bowls and plates
    differ; the open fingers are 10 cm apart.
  - **Stances at the booth:** the base reaches only its north side, where chairs and tables leave little room. A
    stance has the dish 0.48-0.90 m ahead. An offline check found 22-41 arm poses for dishes up to 0.94 m ahead.
- **The sink is known.** Its basin's middle is fixed (`SINK_AT` in `trash_observed.py`). The base stands with it
  0.66-0.76 m ahead, facing north. A dish is let go hanging from the hand pointing down, 8 cm above the deck. An
  offline check found 41-46 arm poses.
- **The kitchen door.**
  - The map is built with the door's leaf and glass pane left out (`--door door_mkcndz_0 --door-swing -5.432
    -3.214 0.84`).
  - A route between the dining room and the kitchen first goes to a point in front of the vestibule. It then
    crosses the doorway (0.93 m wide) on a straight line along its middle, nose first, at 0.15 m/s. Going west, the
    chassis pushes the leaf open, and it stays open (`route()` in `trash_pipeline.py`).
- **Lookouts:** three places north-west of the booth; north of it, chairs and tables leave no room.

### Known limits (k01; see above for what k02 changed)

- **Localisation through the doorway.** On one development run the pose estimate was 1 m off after the return
  through the door.
- **Dishes that miss.** A dish let go over the basin can end on its edge or on the drainboard.
- **Arms that find no pose.** One arm may find no pose above the basin from a stance; its dish then waits for the
  next trip.
