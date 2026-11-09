# c01: loading the car with a map prior and classical motion planning

First controller for this task. It follows the design of the fax, battery and trash controllers (`m05`, `b01`,
`t01`) and reuses their whole-body IK, joint-path planner, base driving, localiser and map format.

## The task

The robot starts somewhere in the garage of `house_double_floor_lower`, next to a car whose trunk is shut. A
toy box (0.305 x 0.455 m, 0.155 m high, 0.56 kg) stands on the living-room floor, twelve metres of driving
away through a door and a corridor; a digital camera and a tennis racket lie on the coffee table there. At the
end the box must be inside the car, the camera inside the box and the racket inside the car. Horizon 28839
steps (1.5 x the human mean of 19226). Each of the three conditions is a third of the Q-score.

What the simulator makes of this, measured (`dev/jobs/probe_*.py`):

- The car has one joint, the trunk lid. "Inside the car" is the trunk: a volume 1.09 m deep, 1.26 m wide and
  0.33 m high above its floor, 0.74 m above the ground.
- The lid (3.1 kg) is heavily damped: let go below 1.9 rad it sinks shut within seconds, above 2.0 rad its
  weight keeps it open. Its lower edge lies sunk in the car's rear when it is shut: there is nothing to pull.
- The task's definition has `not open car` outside its `and`; the evaluator grounds three conditions only and
  ends the episode the moment they hold. The trunk is therefore left open.
- The door between garage and corridor is shut at the start, has no latch and stays where it is left.
- Grasping is the simulator's assisted grasp: when the fingers close with both of them touching a thing that
  lies between them, a joint holds it to the hand until the gripper is commanded fully open. For a link of an
  articulated thing (the lid) it is a ball joint.

## What it does

1. **Localise** in the garage against a static map (`priors/house_map.npz`), with a full turn.
2. **Open the trunk**, with both hands free: stand behind the car, take the pose from the lid's rear edge,
   left hand onto the edge between the lid's top and its rear face, lift to 0.5 rad; right arm alone takes the
   lid's lower end from below; left lets go; right hand leads the lid to 2.1 rad and lets go.
3. **To the living room**: straight through the doorway (the chassis pushes the door open), along the corridor.
4. **Look** for the box, the camera and the racket from the way in, turning on the spot.
5. **Camera into the box**: stand at the table, take the camera by the hump on its top, stand at the box,
   lower the camera into it.
6. **Box to the car**: take the box by the middle of a long wall, hold it level in front of the chest, drive
   back, lift it over the sill and set it on the trunk's floor.
7. **Racket to the car**: back to the table, take the racket by its handle, drive back, hold it flat over the
   box, across the trunk. The episode ends there.

## Why it is built this way

- **Two trips.** With the racket in one hand the other arm found no way down to the box (the trunk joints
  bend for the reach and the racket swings with them), and a hand that holds the box level cannot follow the
  trunk down to the table. One thing in the hands at a time; the way there and back is 2200 steps.
- **The lid is opened by two hands in turn.** With the lid shut, the only place where it can lie between two
  fingers is the edge between its top and its rear face: the hand comes onto it slanting down by 50 degrees.
  Held there, the lid can be followed to about 1 rad (beyond that the hand would have to be on the far side of
  it). So at 0.5 rad the right arm, alone (the trunk joints stay put while the left hand holds the lid), takes
  the lid's lower end, which is free by then, and leads it the rest of the way with its fingers parted: a
  gripper command short of fully open keeps the grasp joint, and the hand's attitude is then free.
  The right hand takes the lower end 1.5 cm inside the lid's rear face, in the middle of its thickness; on the
  face itself one finger closed on air and the joint formed one time in three. When the left hand lets go,
  the right arm sinks 14 cm under the lid's weight: from then on the lid's angle is read off the hand's place
  round the hinge, not counted from the hand-over (counted, the lid was let go at 1.9 rad and sank).
- **Behind the car the pose comes from the car.** Seen from above, the lid's rear edge is a V with a rounded
  point; where that point lies in the base frame gives the pose to 0.2 mm along the car, 7 mm across and 0.1
  degree (`Observer.lid_pose`). With the lid open, the bumper's profile does the same along the car
  (`Observer.bumper_shift`). Between such a fix and the stance the position is kept by dead reckoning.
- **The car is no landmark for the scan match, but it is one for the search for the start pose.** Its
  collision hulls, which the map is made of, are not what the camera sees of its rounded body: a start beside
  the car was fixed 9 cm off. The car is taken out of the fields the pose is tracked with, and what is seen
  where it stands is left out of the match.
- **The garage looks alike from opposite corners**: a bare rectangle, which a half turn about its middle maps
  onto itself but for the car and the door. The start pose is searched for with three things that tell the
  corners apart: the car's hulls stay in the map for the search (a start beside the car has the car in half of
  its view; with those points counted as misses, start 309 was fixed 5.8 m off, in the opposite corner); so
  does the leaf of the door, shut at the start (start 317, next to the door, 6.3 m off without it); and a pose
  is marked down for what it could not have seen: the line from the camera to a point must not run below the
  top of anything in the map's height field (from the corner behind the car, most of what the opposite corner
  sees would lie behind the car). On looks recorded at the 50 development starts (`dev/test_localizer.py`):
  50 of 50 right from three looks, the lead over the best other pose at least 0.13 (before: two wrong, and
  leads down to 0.02).
- **The trunk counts as open only if the base has stood behind the car** and the shut lid's edge no longer
  shows from there. Not seeing the lid from elsewhere says nothing.
- **The lookouts give way to the box.** The box stands within 0.9 m of the first lookout of the living room on
  12 % of the training starts and of the second on 35 %: a turn on the spot there shoved it 0.7 m away. A lookout
  is moved to the nearest place where the base turns clear of the box, and a way is planned again when the box
  first comes into view on it. Routes round the box are followed with a shorter look-ahead (0.20 m), which cuts
  3 cm off a bend instead of 9. A way to a stance at the box keeps the same room round it, and the base gets no
  nearer to the box than the way's start and goal are: planned without that, a way to a point behind the box
  led over it and shoved it 0.35 m. A stance whose staging point has no such way, or no room to turn clear of
  the box, does not count. After a pinch that closed on nothing the base backs off 0.6 m and looks (from the
  staging point of a stance the nearest 0.8 m of floor are below the camera's view) before it goes round the
  lookouts again.
- **The camera is taken by the hump on its top.** Its body is 11 cm thick, more than the fingers open. The
  hump is 7 cm across: the fingers close on it along the body's long side. In the depth image the hump's top
  is the 6.5 x 5.5 cm of points within 9 mm of the camera's highest; the body's long side comes from the
  rectangle round all its points.
- **The box is carried by a long wall, level.** Held by the middle of a long wall it reaches 0.3 m ahead of
  the hand and is no wider than the chassis, so the base turns as freely as without it. Every move of the hand
  with the box keeps the fingers pointing straight down (`level_way`): a planned joint path keeps the hand's
  attitude only at its ends, and on one the box tipped and the camera fell out.
  The trunk is brought upright along that same path: the localiser corrects the pose only with the trunk
  upright, and a box carried with the trunk as the reach had left it was driven into the stairs, 1.3 m off.
  Each hand takes the wall one way round only, and the trunk is not turned about the vertical for the reach: from
  the other way round, and from a turned trunk, there is no level path to the carrying posture (offline, every
  one of 108 wall positions has the whole chain of level paths, into the trunk included, when taken this way).
  A stance counts only if that level path exists from it.
  Where furniture leaves no stance in front of a long wall, the box is taken by a short wall, carried back to
  open floor, set down and taken again.
- **The racket is carried as the hand happens to hold it.** The handle can be taken either way round. One way,
  the racket goes flat above the head, pointing back; the other way the wrist does not turn that far, and the
  racket hangs in front of the chassis, head down, 0.10 m to the arm's side of the middle (nearer to the middle
  the forearm lies against the trunk). Both are no wider than the chassis and inside the circle it sweeps.
- **What a hand holds is part of the robot** for the collision checks (`Context.payload`): spheres for the box,
  the racket and the camera move with the tool frame, and are cut out of the scans as well.
- **A last look with the wrist camera before each pinch.** After a long way in, what the head camera saw of a
  thing is 1-3 cm off, and the fingers have 1 cm to spare on the camera's hump: from one stance the hand closed
  on nothing three times. With the hand above the thing, the wrist camera's depth image gives the hump, the
  racket's handle or the box's wall to a few millimetres, and the hand is put right before it goes down.
- **Stances next to the table are found with the chassis's rectangle**, since the route planner's disc has no
  room there; the base turns at a staging point and moves in straight (sideways into the gap between the sofa
  and the table, where the chassis has 3 cm to spare and one distance only fits). A stance counts if the arm
  planner finds the whole pinch from it.
- **In the trunk the map is a height field**: further in than x = -6.33 it has the parcel shelf's height and no
  room under it. The box is set down with its middle at x = -6.50 (it counts as inside from -6.60), and the
  racket lies across the trunk, not along it. The hand leaves the box the way it came (straight up first): an
  arm that folded where it stood dragged the box out again.
- **After the door, the base turns in the middle of the corridor.** A turn next to the door swept the leaf with
  the chassis's rear and shoved the base 10 cm along the corridor, where the scan match cannot tell.

## Known limits

- A box that stands in the 0.7 m gap between the sofa and the coffee table, partly under the table's edge, has
  no stance: neither for lowering the camera into it nor for taking it (public start 313; 1 of the 280 training
  starts that are not held out). Only the racket is loaded then.
- If the box cannot be put into the trunk (two tries), the hand keeps it and the racket is still fetched.
- If the trunk does not open in three tries, the rest of the task is still run, to no effect.
- Offline, with the true places of the things (`dev/study_stances.py`): of 280 starts, a stance for the camera
  and for the racket exists on all, for the box on all but one; two need the box taken by a short wall first.

## Inputs and priors

Runtime inputs: the head camera's depth image, the wrist cameras' depth images (for the last look before a
pinch), 61-D proprioception, the cameras' poses relative to the base (the evaluator supplies them; they equal
forward kinematics of the proprioception on the robot model to 0.0 mm), the policy's own last action. RGB is
not used.

Priors, the same for every instance: the robot model; the map of the house's lower floor with the car and the
coffee table, built once from the scene's collision geometry (`dev/build_scene_map.py`); the sizes of the toy
box, the camera and the racket; the geometry of the car's rear (hinge line, the lid's edges, the profiles of the
lid's rear edge and of the bumper as the depth camera shows them, the trunk's floor and extent). In the source
besides: the line through the doorway, the lookouts of the living room, the odometry model measured for the
battery task.

No simulator state, segmentation, contacts or evaluator results are read.

## Files

| File | Role |
|---|---|
| `observation_policy.py` | Evaluator entry point (`Policy`) |
| `car_observed.py` | Estimator fed by the camera (the things, the pose fixes at the car), per-step runtime, global localisation |
| `car_pipeline.py` | The task as a sequence of skills: the trunk, the door, stances, the things, the two trips |
| `car_skills.py` | Opening the lid, pinches from above, setting down, carrying, shapes of the things for the collision model |
| `observer.py` | The box, the camera, the racket, the lid's edge and the bumper in the depth image; localisation scans |
| `manip.py` | Context with the collision validator (robot and what it holds), base driving, joint-path planning |
| `localizer.py`, `scene_map.py` | Scan matching in height slabs, dead-reckoning corrections; map prior, routes, clearances |
| `wbik.py`, `skills.py`, `kinematics.py`, `robot_collision.py` | Whole-body IK (also position only) and joint-motion primitives |
