# b01: dispose of batteries with a map prior and classical motion planning

Development candidate, not promoted. First controller for this task; it follows the design of the
fax controller m05 (`tasks/installing_a_fax_machine/experiments/candidate/m05`) and reuses its
localiser, whole-body IK, joint-path planner and base driving.

## The task

Three batteries (standing cylinders, 3.4 cm by 6.1 cm, 75 g) must end up inside the trash can, and the
can must stand on the copy-room floor. One battery stands on the copy-room cabinet (1.02 m high), where
the robot and the can start; two stand on the desk of a private office 14 m to the north, behind a
closed door. Horizon 21 642 steps. Each battery in the can is a quarter of the Q-score.

## What it does

1. **Localise** in the copy room against a static map (`priors/office_map.npz`), with a full turn.
2. **Look around**: the can is a knee-high object on floor the map has empty; batteries are blue caps
   whose 3-D position lies on a known work surface.
3. **Cabinet battery**: stand at the cabinet, pinch the battery, drop it into the can.
4. **Move the can** if it stands in the way of that or of the door (on a third of the starts it does):
   pinch one of its walls, lift it, translate without turning, put it down where the exit stays
   passable.
5. **Go to the office**: out of the doorway, 12.5 m up the corridor, through the door. The door opens
   into the office and has no latch: the chassis pushes it open. The robot passes sideways.
6. **Desk batteries**, one per hand. Those within 0.4 m of the door-side edge are taken from there. The
   others cannot be reached from that side, so the robot drives past the east end of the desk to its
   far side, gets the office chair out of the way and takes them from there.
7. **Return** the same way and drop both batteries into the can.

## Why it is built this way

- **Reach.** With the static load on the first trunk joint limited to 85% of its effort limit, the far
  0.3-0.4 m of the 0.93 m deep desk are out of reach from the door side (`dev/study_reach.py`); 70% of
  the public desk batteries lie there. From the far side every battery is within 0.4 m of the edge.
- **The passage past the desk** is 0.52 m wide at desk height and 1.08 m at floor level. The chassis
  (0.68 m wide, 0.41 m tall) runs under the desk top; the folded trunk passes 0.03 m from its edge and
  the chassis 0.04 m from the wall. Measured: passable on x = 11.30 and 11.34, not on 11.38
  (`dev/jobs/slot_trial.py`). It is driven front first in both directions, so that the end of the desk
  top is in view on the way in (see localisation below).
- **Posture.** Trunk upright and both arms folded in front of the chest (`TUCK` in `manip.py`): nothing
  wider than the shoulders and nothing below 0.96 m, so the arms clear the desk top in the passage.
- **The office chair** stands where the robot has to stand. Two ways of clearing it. For each, stances
  and arms are planned for the batteries on the far half (each battery needs an arm that reaches it on
  its own side: reaches across the body were planned and then refused at the desk), and the way that
  serves more batteries, then the sturdier reaches, is taken (`chair_mode` event):
  - *into the knee hole*, pushed from behind along x = 9.96. The robot then stands on either side of it.
    In 18 development runs of 18 the chair's hub ended at x = 9.66-9.72 leaning on the desk; the stance
    zones and the box the arms keep clear of are set from that;
  - *to the west wall*, pushed with the **side** of the chassis while the robot faces the desk and moves
    sideways along it on y = 4.86. The wheels are under the two front corners and under the middle of the
    rear. Pushed with the front, the feet of the chair's star base get under the front wheels: the chair
    tips 9-16 deg, the robot leans up to 6 deg and drags the chair along when it leaves. The side has no
    wheel behind the front corner; on y = 4.83-4.90 the chair rolls away upright and stays, on y = 4.98
    a wheel climbs a foot and the robot leans 28 deg (`dev/jobs/chair_west_trial.py`).
- **Batteries fall over at a touch**, and lying down they cannot be pinched. The arm's way to and from
  a grasp is planned in joint space against the furniture; at a lateral reach of 0.42 m it swept through
  the battery it was about to take. Every battery known to stand on the surface gets a guard box in the
  collision check (`Context.guard`), the one being taken until the hand is above it.
- **Many batteries stand at the very edge of the desk** (1 cm from it). The hand normally comes in along
  the line from the robot and closes across it; for a battery 0.28 m to the side the finger that touched
  first then pushed it over the edge onto the floor. A battery within 7 cm of the edge the robot faces
  is pinched with the fingers closing along the edge (`plan_reach(square=True)`).
- **The can's wall** is pinched straight ahead of the robot, where each arm reaches it with the fingers
  closing one way only. Both arms and both ways are tried (`plan_can_lift`). The route past the can
  keeps the base 0.64 m from it, less only when the room leaves no other way.
- **Localisation where centimetres matter.**
  - In the corridor and in the copy room: scan matching against the map. The map has separate layers
    for what the depth camera sees (a window is its pane, not the solid box of its frame's collision
    hull, which stands 7 cm in front of the pane). A coordinate the view does not constrain (a plain
    wall straight ahead) is not corrected.
  - Through the office door: the end of the wall the door is in is measured directly, then the position
    is dead reckoned while the robot slides in; the swinging door leaf is masked out.
  - **In the office the pose comes from the edges of the desk top** (`Observer.table_edge`,
    `ObservedEstimator.follow_edge`), from the moment the robot faces the desk until it is back in the
    corridor. Scan matching is unreliable there: behind the desk it was off by 10-15 cm and 2-3 deg
    (little of the map is in view, pushed chairs are), which wedged the robot in the passage and once
    tipped it over; on the way out it was 10 cm off in a doorway that leaves 5 cm to either side. The
    edge of the top is in plain view from a metre away: on captured views with known poses it gives the
    coordinate across the edge to 1-7 mm and the heading to 0.0-0.3 deg (`dev/jobs/capture_views.py`).
    The top as seen ends 1.5 cm inside the box the map has for it. Where an end of the edge is in view
    (the desk's east corner, from either side) the coordinate along the edge is read too; otherwise it
    is dead reckoned. Closer than about 0.45 m the edge is below the image and no reading is taken.
  - **Dead reckoning** follows the velocity command, not the base's own report of its translation
    (`dev/jobs/odom_calib.py`, 47 legs in free motion): the base moves at 0.75 m/s per unit of command
    less a fixed 6.4 mm/s forwards and 2.3 mm/s sideways, within 1.2 deg of the commanded direction, and
    turns 0.03 rad/s slower than commanded on the spot (0.01 while driving). Replayed on those legs this
    ends 5-8 mm off after 0.5-1.5 m. The report is good to a few per cent forwards only; sideways it read
    between 36% short and 12% long depending on what the base had done before. It is used to notice that
    the base is held up (it then reads next to nothing) and while the base coasts to a stop.
- **Grasp targets are measured in the base frame** when the hand is about to move (head camera now, or
  the last look carried by odometry), so an error of the map pose does not enter the grasp. A known
  battery is recognised up to 0.25 m from where the map pose puts it. A tilted approach aims 8 mm past
  the battery's axis: stopping short pushes the battery away. With the hand in front of the battery the
  wrist camera looks at it: a steep approach always takes its reading; a tilted one takes it when it
  disagrees with the target by more than 2.5 cm (a battery straight ahead is hidden from the head camera
  by the folded hands, and its last sighting, carried past a pushed chair, was once 10 cm off).

## Inputs and priors

Runtime: head and wrist RGB-D, 61-D proprioception, robot-relative camera poses, the policy's own
last action. Static: `priors/office_map.npz`, `r1pro.urdf`, `robot_model.json`,
`robot_collision_spheres.json`.

The map is built once from the scene's collision geometry by `dev/build_scene_map.py` (input: a dump
made by `dev/jobs/probe_static.py`), from assets 3.9.0. It is the same for every instance of this
task because the scene is. It also carries fixed facts of the scene used by the script: the
footprints of the desk, the cabinet and the copy-room floor, and where the office door swings.
The script itself contains scene-specific places and lines (the corridor, the doorway, the passage,
where the chair is pushed). Whether a scene map and such a script are admissible is a rules decision.

## Files

| File | Role |
|---|---|
| `observation_policy.py` | Evaluator entry point (`Policy`) |
| `battery_observed.py` | Estimator fed by the cameras, edge following, per-step runtime, looking around |
| `battery_pipeline.py` | The task as a sequence of skills, driven by an estimator (camera-based or, in dev jobs, simulator truth) |
| `battery_skills.py` | Reach planning, grasp, fold, drop, lifting the can |
| `manip.py` | Context with the collision validator, base driving, joint-path planning |
| `observer.py` | Battery caps, the can, table edges, the wall end at the door, localisation scans |
| `localizer.py`, `scene_map.py` | Scan matching and tracking, dead-reckoning corrections; map prior, routes, clearances |
| `wbik.py`, `skills.py`, `kinematics.py`, `robot_collision.py` | Whole-body IK and joint-motion primitives (from m05) |

## Known limits

- Two desk batteries on the far half, one behind the chair in the knee hole and one within 0.4 m of the
  desk's west end, are both reachable only if the first can be had from beside the parked chair at a
  lateral reach of 0.55 m.
- A battery within 3 cm of the cabinet's end walls gets no stance (4 of 280 development starts).
- The can's carry-away plan fails on about 4 of the 96 development starts where the can is in the way
  (`dev/study_can.py`): the robot starting on top of the can, or the can in the doorway corner.
- A battery that has fallen over is not picked up. Nothing checks that a dropped battery landed in the can.
- The can is recognised by its size and place only; knocked over it is not found again. Its heading is
  read from the smallest box around its rim when most of the rim is in view; it is looked for up to
  0.35 m outside the copy-room floor (it can stand in the doorway).
- Dead reckoning assumes free motion. Pushing a chair forwards is noticed from the base's report and
  followed; pushing one sideways is not (the report is no use sideways), and the position along the desk
  is then read again from the desk's corner.
