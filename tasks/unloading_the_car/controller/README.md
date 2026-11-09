# u01: unloading the car with a map prior and classical motion planning

First controller for this task. It is built on the controller for `loading_the_car` (`c01`): the same house, car
and door, the same localisation, opening of the trunk lid, way through the door, routes and arm planner. What
is new is taking two bags out of the trunk and putting them down at the sofa.

## The task

The robot starts somewhere in the garage. A briefcase and a satchel lie in the car's trunk, which is shut. At the
end both must be next to the sofa in the living room, twelve metres of driving away through a door and a
corridor. Horizon 16985 steps (1.5 x the human mean of 11323).

What the simulator makes of this, measured (`dev/jobs/probe_static.py`, `probe_bags.py`) or read in its source:

- Both bags weigh 1.5 kg and are single rigid bodies. The briefcase is a slab, 0.23 x 0.30 m and 0.089 m thick,
  with a small handle on one side that is of no use to a hand. The satchel is a wedge 0.41 x 0.49 m and 0.22 m
  high: a low sloping front, a flat top 9 cm wide, a back that falls away at 71 degrees, upright ends. Neither
  fits between the fingers, which open 10 cm.
- The satchel lies upright on the trunk's floor, its middle 0.35 to 0.55 m in from the sill, turned any way. The
  briefcase lies flat on three starts in four; on the rest it leans on the satchel or has a side on the step of
  a wheel arch (up to standing on a rim, or upside down). They do not move when the lid opens.
- The trunk: its floor is 0.76 m up at the sill and slopes down 4 degrees to the seat back; the opening is
  0.47 m long and 1.1 m wide; behind it the parcel shelf leaves 0.36 to 0.39 m of height. What the camera shows
  of the floor (the carpet) lies 3 to 6 cm below the surface the bags rest on.
- "Next to" (`object_states/next_to.py`): the gap between the two bounding boxes is at most one eighteenth of
  the sum of their six sides (0.36 m for the briefcase, 0.39 m for the satchel), and a level ray from the middle
  of one of them, in one of sixteen directions, hits the other. A bag on the floor with its box within 0.3 m
  east of the sofa's chaise has both; 0.47 m away it has not (`probe_bags.py`).
- The episode ends the moment both bags are next to the sofa, also while the second is still in the hand.

## What it does

1. **Localise** in the garage and **open the trunk** (as c01).
2. **Open the door** between garage and corridor with empty hands: drive through (the chassis pushes the leaf
   open; it stays where it is left) and come back.
3. For each bag, the satchel first (the briefcase first if it lies on the satchel or in its way out):
   - to the staging point behind the car, the pose taken from the bumper; **look into the trunk**, with both
     bags there also from the stance on the car's middle line;
   - stand behind the car where a shoulder faces the edge the hand is going to take; look again;
   - **take the bag by an edge** with a slanted hand; lift it 11 cm, draw it back 22 cm from under the parcel
     shelf, lift it over the sill;
   - back to the staging point; hold the bag in front of the chest, hanging from the hand;
   - through the door and the corridor to the stance east of the sofa's chaise; reach forward, **lower the
     bag** to the floor, let go, draw the hand back along its fingers.
   A visit that does not get its bag (the hand closed on nothing twice, or found no way) is followed by one for
   the other bag, then by another for this one from another edge or with the other hand; eight visits at most.

## Why it is built this way

- **A bag is taken by an edge, the way the trunk lid is.** The fingers point forward and down by 50 degrees; one
  finger lies on the bag's top face, the other on the side face that looks at the robot, the edge between them.
  The simulator's assisted grasp then holds the bag (both fingers touch it and it lies between them), and it
  stays rigid in the hand. The point between the fingers' ends goes 2 cm below the top face and 1.5 cm inside
  the side face: the upper finger then comes in 1 cm above the top and the lower one 2 cm outside the side. With
  the numbers of the trunk lid (4.5 cm below the edge) the upper finger came down on the bag, the hand shoved it
  and closed on nothing, in 9 tries of 12.
- **The satchel's edges are those of its real shape.** Its back ridge (0.134 m from its middle, 0.105 up) and
  the top edges of its two ends, measured on its collision shape. Taken for a box of its size, the edges were 1
  to 2 cm further out and lower than they are: 9 of 22 tries held; with the measured ones, 18 of 26. On one end
  a strap ring stands out below the edge: the hand goes on beside it and less deep (before: 2 of 8 on that end).
  A satchel whose low front looks at the robot has its back and the high part of its ends under the parcel
  shelf, out of reach: it is taken by the low part of an end, the hand pointing sideways. With all of this,
  and the two points below, 19 of 19 on the starts that had failed.
- **The hand that points left is the right hand.** A left arm that reached round to point left shoved the bag
  away; the arm is chosen by the direction the hand points along.
- **The bags are obstacles for the arm** (`Context.obstacles`): each as spheres on its surface, where the look
  into the trunk has it. The working hand alone may touch the bag it takes. Without this the planner took
  postures in which the forearm swept through the satchel.
- **The satchel first**, since a briefcase that leans on it then comes to lie flat; unless the briefcase lies
  between the satchel and the sill, where the satchel is drawn out: it swept a briefcase out of the trunk that
  lay flat there, and two that leant on it from there (0.13 and 0.17 m nearer to the sill than the satchel),
  onto the garage floor. A briefcase 0.12 m nearer to the sill (0.10 m if it leans) goes first, and is tried on
  four visits, by other edges and the other hand each time, before the satchel is. The satchel is lifted 11 cm
  before it is drawn back: over a briefcase lying flat, and 2 cm under the parcel shelf.
- **A briefcase that lies on top of the satchel goes first** (6 of 280 starts): drawn out from under it, the
  satchel took it along and it fell to the floor behind the bumper (start 34). The hand that takes it may come
  as near to the satchel as to the briefcase.
- **The order is chosen after a look from the stance on the car's middle line**, nearer and from above: a
  briefcase that leans on the satchel does not always show from the staging point.
- **Where the bags lie comes from the head camera's depth image, matched to their shapes.** Points inside the
  trunk and above its floor are matched to the height of each bag's upper surface (`priors/bags.npz`, from the
  collision hulls): the place and heading with most points on that surface, inside the trunk's walls. The
  satchel first; the briefcase on what the satchel does not explain. A briefcase that does not match well
  lying flat (it leans, stands on a side, lies upside down, or lies against a wall half out of view) is found
  as a slab: the plane with most points is one of its faces, broad or narrow, and every way of laying the slab
  against that plane is held against all the points round it: those on the slab's surface count for it, those
  that would lie inside it against it, and its lowest corner must lie on the trunk's floor, the sill or the
  satchel. On 381 frames recorded with true poses, from probes and from the looks of development runs
  (`dev/test_trunk_view.py`): satchel 1.0 cm median over 271 fits; briefcase lying flat 0.9 cm over 94;
  leaning or standing 1.5 cm over 214. Errors above 3.6 cm only on three starts where the briefcase lies on
  the satchel: the briefcase up to 10 cm, the satchel under it up to 13 cm (both bags came out there all the
  same, on later tries).
- **The satchel and the briefcase are chosen together.** With a briefcase leaning on it the satchel matched
  as well turned by 60 degrees and 12 cm off as where it was (start 103), and the hand closed on nothing twice.
  Up to three ways of laying the satchel are each taken with the briefcase that the left-over points show; the
  pair that explains most points is the one (0.7 cm on those frames).
- **It is a bag's outline that must lie inside the trunk's walls, not the box round it.** A satchel that lies
  askew against a wall has a corner of its box up to 7 cm beyond the wall (one start in eight): with the box,
  its true place was ruled out, it was matched 7 cm off or not at all, and on one start neither bag was found and
  nothing was taken.
- **The plane of the briefcase's face is cut to one face's size** before its outline is measured: a few points
  elsewhere in the same plane made the outline too long for a briefcase (not found in 83 of 448 tries over the
  recorded frames; 2 of 448 after). A plane that covers less than 0.45 of a face is not taken for it.
- **A look that does not show a bag leaves what the look before it showed**, and a bag that does not show from
  the staging point is looked for from the stance on the car's middle line. A bag that has been taken out is
  not looked for: what the satchel's shape left unexplained was taken for the briefcase on 3 looks of 380 after
  the briefcase had gone.
- **An edge is remembered by which edge of the bag it is**, not by its place among the edges ordered by
  nearness: that order changed between the look from the staging point and the nearer one, and the hand was
  sent to another edge than the stance was for. A hand that finds no way to the edge from where the base came
  to stand goes on from another stance (three at most) without driving back.
- **A briefcase that leans or stands is taken by an edge that has a face looking up**, for the upper finger,
  which then presses the briefcase onto what carries it, as on a bag that lies flat. The pinch is the same for
  any two faces (`bags.pinch`); the candidates are all such edges of the slab (`bags.edges`), nearest first.
  Leaning on the satchel: its upper rim from above, the lower finger on the broad face that looks up. Standing
  on a long side against the trunk's wall (start 31): the near end of the narrow face on top.
- **On that rim the fingers go 2.6 cm inside the broad face**, not 1.5 cm as on a bag that lies flat: the lower
  finger then starts 0.5-1 cm from the face and touches it first, pressing the briefcase against what it leans
  on, before the upper finger comes down on the rim. With 1.5 cm the upper finger came first and pushed the
  briefcase down and away along its own plane: the hand closed on nothing in 5 tries of 5; with 3 cm inside the
  true face (privileged trial) it held 2 of 2 on the same starts.
- **A satchel with its low front to the robot and an end at the trunk's wall is taken with the hand askew.**
  On 8 of 280 development starts no edge had a stance: the back under the parcel shelf, one end against the
  wall, the other looking away from the robot (found by running the stance planner over all starts offline,
  `dev/plan_all.py`; none of the 50 starts used in the simulator had it). The hand is turned by 35 degrees
  about the line its fingers close along, so that it points more ahead; the fingers close across the edge as
  before. The planner then has a stance on all 8. In the privileged trial the pinch held on 4 of the 8: on the
  end towards the trunk's middle 2 of 2; on the end by the wall 2 of 5 (on the others the hand was stopped 6 to
  8 cm short on its way in). So the end with more room to the walls is tried first.
- **A hand that is stopped more than 3 cm short of the edge does not close**; it goes out the way it came and
  takes another edge. An edge and hand that did not get the bag come after all the others on later tries.
- **The trunk is looked into over the sill as well.** On start 216 the briefcase was left lying across the
  sill, half outside, when the satchel was drawn out past it; with the camera's points cut off at the sill it
  was taken to lie 17 cm further in, and the hand closed on nothing six times. For a bag on the sill the base
  stands 0.22 m further from the car.
- **The trunk is described by the car's collision hulls**, not by the map's height field, which has the parcel
  shelf's height over the trunk and no room under it (`dev/add_trunk_cavity.py`, `SceneMap.hulls`). A hand from
  above reaches 0.35 m in from the sill; further in the forearm meets the edge of the shelf. So a bag is taken
  by its near side and drawn back.
- **The door is opened before any bag is carried.** A bag in front of the chest would meet the leaf before the
  chassis does.
- **The bag is carried as the hand holds it**, turned down so that it hangs ahead of and below the hand, in
  front of the chest and no wider than 0.36 m to a side (the doorway is 0.41 at chassis height). The posture is
  searched for with the bag in the collision model, since it depends on the bag and the edge it is held by.
  It is taken at the staging point: next to the car the bag would be in the bumper.
- **At the sofa the bag is put down 0.7 m ahead of the base**, by reaching forward first, and the hand draws
  back along its fingers before the arm folds. Let down close to the chassis and with the arm folded at once,
  the satchel was dragged 0.4 m back from the sofa by the hand, which still had its fingers astride its edge.

- **Once the car's lid shows where the map pose has it, the other start poses are dropped** (`car_observed.
  fix_at_lid`). c01's localiser follows look-alike start poses for a while (one start in ten has one) and lets
  one take over if it explains the views better. On start 152 the pose in the garage's opposite corner took
  over 50 steps after the pose fix at the lid, with the camera full of the car's rear and the true pose
  matching at 0.99; the robot drove by a pose 5 m and half a turn off into the wall.
- **One search for a stance is cut off after 40 s**, and what the planner had nothing for is remembered from
  visit to visit while the bag lies where it lay: a satchel with no stance held up one step of the policy for
  100 s. Every edge is tried with its own hand at the stance that faces it before any stance to the side.

## Known limits

- A bag that leaves the trunk other than in the hand (swept out by the other bag, onto the garage floor) is
  not looked for and not picked up; the episode then ends with one bag at the sofa.
- A briefcase that leans steeply on the satchel from the sill side is the hardest to take: in development the
  hand closed on nothing on up to four tries before it held, and on one run the satchel was taken after two
  visits and swept it out. Four visits are now made before the satchel is taken.
- Nothing checks that a bag stays in the hand on the way, or that it lies next to the sofa after it is let
  down. If the arm finds no way down at the sofa the bag is let go from where it hangs (seen in the garage
  only, in runs of the trunk part).
- One step of the policy may take up to 40 s (the search for a stance).

## Inputs and priors

Runtime inputs: the head camera's depth image, 61-D proprioception, the cameras' poses relative to the base, the
policy's own last action. RGB and the wrist cameras are not used.

Priors, the same for every instance: the robot model; the map of the house's lower floor (c01's), with the car's
rear end as convex hulls; the shapes of the two bags (`priors/bags.npz`); the geometry of the car's rear and of
the trunk; the stance at the sofa; the line through the doorway.

No simulator state, segmentation, contacts or evaluator results are read.

## Files

| File | Role |
|---|---|
| `observation_policy.py` | Evaluator entry point (`Policy`) |
| `unload_observed.py` | Estimator with what is known of the bags; entry point of the script |
| `unload_pipeline.py` | The task as a sequence of skills: door, car, the bags, the sofa |
| `bags.py` | The slanted pinch of an edge, the edges of each bag, lifting out, carrying |
| `trunk_view.py` | The bags in the trunk from the depth image |
| `car_observed.py`, `car_pipeline.py`, `car_skills.py`, `observer.py` | From c01: localisation and pose fixes at the car, the trunk lid, the door, routes |
| `manip.py`, `localizer.py`, `scene_map.py`, `wbik.py`, `skills.py`, `kinematics.py`, `robot_collision.py` | From c01, with obstacles in the collision check and hulls in the map |
