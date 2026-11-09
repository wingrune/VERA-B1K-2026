# m05: installing a fax machine with a map prior and a rebuilt manipulation chain

Development candidate, not promoted. Successor of m01 to m04 (same design, see the lists of
changes at the end). It replaces the v24 lineage's controller for this task; only the robot model, the URDF
kinematics and the fax geometry prior are reused from it.

## What it does

1. **Localise** in a static map of the office (`priors/office_map.npz`). Two horizontal bands
   are cut out of the head depth image, one below the cubicle glass and one above the partitions
   where the outer walls show, and matched against the map. The start pose is found by scoring
   every free position and heading; look-alike aisles are kept as rival hypotheses and dropped as
   the views separate them.
2. **Find the fax**: anything a few centimetres tall standing on floor that is empty in the map.
   If it is not in sight after a full turn, the east end of each aisle is visited in turn.
3. **Fit its pose** from about a metre away with the head camera (3-D model against the points
   above the floor; the toggle marker fixes the orientation).
4. **Press the button while the fax is still on the floor**, then check that the marker turned
   from red to green with the wrist camera; press again if it did not.
5. **Pick it up** with a top-down pinch across the raised back step of its body, with whichever
   arm reaches from a stance the map allows.
6. **Carry** it at chest height, drive into a cubicle nook on the west column (pushing the
   castor chair aside) and **lower it onto the desk**.
7. If the floor press never succeeded, press the button on the desk.

## Why it is built this way

- The toggle state persists, and the task score counts each goal predicate. Pressing on the
  floor uses the same arm posture as the grasp; pressing inside the cubicle afterwards was the
  step the earlier controllers never reached.
- The simulator uses assisted grasping: an object is attached when the gripper is closing, two
  fingers of that hand touch it and a ray between the fingers hits it, for 0.3 s. The fax body is
  11 cm wide where the gripper opens 9.8 cm, so the fingers straddle the step on its back instead:
  one finger behind the back wall, the other on the sloping front (`GRASP` in `fax_skills.py`).
- Reaching the floor leans the upper body 0.6-0.9 m ahead of the base while every camera looks
  down. Collision checks therefore use the map (it also holds the glass wall and the cubicle
  glass), not the current camera view.
- In instance 302 the base cannot pass the fax in the 1.2 m aisle, and from the side it can stand
  on only the right arm reaches. Stance planning chooses the arm.
- Proprioceptive base yaw rate is biased by up to 0.14 rad/s; commanded motion is used for
  prediction and scan matching for correction. A correction may move the pose only by a fraction
  of the distance driven since the last one, and never into furniture; without these two rules
  the estimate walked sideways along parallel walls.
- The fax estimate is stored relative to dead-reckoned odometry, not to the map pose, so a map
  correction between looking and touching cannot shift the contact point. Right before contact
  the wrist camera aligns the model's button with the marker it sees.

## Inputs and priors

Runtime: head and wrist RGB-D, 61-D proprioception, robot-relative camera poses, the policy's
own last action. Static: `priors/office_map.npz`, `priors/facsimile`, `r1pro.urdf`,
`robot_model.json`, `robot_collision_spheres.json`.

The map is built once from the scene's collision geometry by `dev/build_scene_map.py` (input:
a dump made by `dev/jobs/probe_static.py`). It is the same for every instance of this task
because the scene is. It is built from assets 3.9.0 and has to be rebuilt for any other asset
version: a controller that passed ten of ten starts on 3.7.2rc1 lost two on 3.9.0.

## Files

| File | Role |
|---|---|
| `observation_policy.py` | Evaluator entry point (`Policy`) |
| `fax_observed.py` | Observation-based task script, estimator, per-step runtime |
| `observer.py` | Fax detection, pose fit, marker reading |
| `localizer.py` | Scan matching, global search, tracking |
| `scene_map.py` | Map prior: free space, routes, height field |
| `fax_skills.py` | Stance planning, press, grasp, carry, place, driving, joint-path planner |
| `wbik.py`, `skills.py` | Whole-body IK (trunk + one arm), joint-motion primitives |
| `fax_pipeline.py` | The same task script driven by simulator truth (diagnostics only) |

## Development tools (`../../../dev`)

`dev_server.py` keeps one simulator alive and runs job scripts from a queue; `jobs/` holds the
probes and the privileged and observation-only trial jobs; `test_localizer.py` and
`test_observer.py` replay recordings with ground truth. Jobs that read simulator truth say so in
their header and are never a scored run.

## Changes from m01

- The approach to the fax (for the press and for the grasp) is planned from the contact pose
  upwards and checked against the map along the way (`plan_descent`). m01 solved the pose above
  the fax first; the arm configuration it picked could not always come straight down, and the
  descent itself was not collision-checked.
- Stance planning prefers base poses with room in the lean direction. In the one-metre aisles
  m01 could choose a lean towards a partition; the shoulders and the idle arm then caught on it
  and the hand stopped six centimetres above the fax (instance 315).
- Modules copied from v24 that the controller never imported were removed.

## Changes from m02

Both come from the first run on assets 3.9.0, where frozen m01 failed two of ten starts.

- Fax candidates are accepted only on the floor the task definition puts the fax on (the shared
  office). On 303 something on the corridor floor was taken for the fax.
- A candidate that fails the close look is remembered and the search continues; m01 and m02
  gave up at that point.
- The press on the desk (the fallback when the floor press was not possible) aimed 1.2 cm too
  high, because the fax settles onto the desk when released. It now aims at the settled fax and a
  little deeper, and contact stops the fingers (instance 305: fax on the desk, button never touched).
- `priors/office_map.npz` is built from assets 3.9.0.

## Changes from m03

Both come from the scored run of m03 on assets 3.9.0 (18 of 20; 308 and 309 failed).

- Localisation ignored nothing: a scan point that fell outside the map counted as a miss. The
  map ended 0.6 m south of the office while the corridor runs five metres further, so from the
  south aisle facing south the true pose scored 0.43 and its look-alike in the middle aisle 0.73.
  Points off the map are now left out of the score, and the map covers the whole corridor and
  the neighbouring rooms. (Starts 306 and 308; 308 never recovered.)
- The move of the fax over the desk was planned without the chair (it has been pushed and its
  place is unknown) and was sometimes stopped by it. The arm and the fax now stay above
  chair-back height in front of the base, and when the arm is stopped another spot on the desk
  is tried instead of giving up (start 309: fax held, switched on, never put down).

## Changes from m04

All come from the second scored pass of frozen m04 on assets 3.9.0 (19 of 20; the first pass was
20 of 20). Start 312 failed: at the floor stance the grasp was found out of reach twice and the
script gave up with four thousand steps left. In the first pass the same check had failed once
and passed on the second try.

- The lean-room term of stance planning (m02) did not work. It probes the space ahead of the base
  with spheres of up to 0.46 m, but the map's clearance function looked only 0.22 m around a
  sphere's centre, so every candidate got full marks. `SceneMap.sphere_clearance` now covers the
  sphere it is asked about. On 312 the fax lies at the mouth of the one-metre part of the aisle;
  the stance chosen without that term leaned the upper body into the end of a partition, 23 of
  24 arm configurations were rejected for the idle arm's shoulder, and the one that fitted (torso
  twisted away) was lost when the base stopped 3.4 cm short. With the term working the stance
  faces along the aisle and 13 of 14 configurations fit.
- Stance planning counts the collision-free arm configurations that reach the grasp and takes a
  stance at once only if there are at least three; a stance with fewer is used only when nothing
  better is found.
- The last straight stretch to the stance ends within 1.5 cm of the planned pose instead of 3.5 cm.
  The drive stopped as soon as it entered the tolerance, so the base was always that much further
  from the fax than planned.
- At the stance, when no approach is found, the search is repeated once with three times as many
  seeds before the stance is given up.
- A stance that was not reached or from which the grasp was out of reach is not offered again, and
  the floor is tried up to five times instead of two. A failed drive to a stance no longer ends
  the script. The button is not pressed again on a later try once the marker has been seen green.

## Known limits

- Placement uses only the five west-column cubicles; the east-column nooks are closed on both
  sides and the chair jams there.
- The middle aisle is 1.0 m wide and leaves the base a strip of about 0.2 m to stand in; the
  stance and approach checks above exist because of it.
- The search visits aisle ends in order of distance; it does not yet use what it sees on the way.
