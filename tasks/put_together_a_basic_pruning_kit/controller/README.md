# p01: putting the pruning kit together with a map prior and classical motion planning

First controller for this task. It starts from the controllers for `loading_the_car` and `unloading_the_car`
(`c01`, `u01`): the same house and garage, the same localisation, routes and arm planner. What is new is finding
three things on the garage floor, taking a pruner and a pair of shears from the floor, and working the lid of a
toolbox with one hand while the other lets a tool go into it.

## The task

The robot starts somewhere in the garage. A toolbox, a pruner and a pair of shears stand on the garage floor,
anywhere on it and turned any way. At the end both tools must be inside the toolbox, the toolbox on the floor
and not open. Horizon 16896 steps (1.5 x the human mean of 11264).

What the simulator makes of this, measured (`dev/jobs/probe_static.py`, `probe_kit.py`, `kit_trial.py`) or read in
its source:

- The toolbox weighs 2.0 kg, is 0.24 m across, 0.62 m long and 0.28 m high to its lid's top. The lid turns about
  a hinge line along one long side, from 0 (shut) to 2.97 rad. It counts as open from 0.148 rad. Let go below
  about 0.9 rad it falls shut; from about 1.5 rad on it falls open and stays open. On top of the lid is a
  handle: a bar 14 cm long and 2.2 cm thick, 4 cm above the lid, along the box.
- The hinge side and the front are the same in depth. The front carries two black latches; the hinge side is plain.
- The pruner stands on the ends of its two handles, blades up: 0.23 m high, 6.7 cm by 2.9 cm across at its
  handles, 0.72 kg. Its blades lean 6 degrees towards one end of the pair of handles.
- The shears lie flat: 13 cm long, 5 cm across at their ring handles, 2.5 cm high, 0.26 kg.
- A tool counts as inside when the middle of its bounding box is inside the toolbox's bounding box and inside the
  hull of its container volume. Let go over the open box in any attitude it comes to lie inside, and the lid
  shuts over it.
- The gripper opens 10 cm; the free depth between its fingers is 7 cm. The head camera does not show the floor
  nearer than 0.86 m ahead of the base: a thing the hand is about to take is out of its view.

## What it does

1. **Localise** in the garage (as c01) and **find the three things**: a look round on the spot, then from up to
   five lookouts until all three have shown.
2. **Learn which long side of the toolbox is its front**: a look at a long side from 1.1 m in front of it.
3. For each tool, the nearer one first:
   - **choose the stance at the toolbox** for the visit that will follow, and with it the hand that opens the
     lid there; the tool goes into the other hand (of two possible hands, the one with the sound way onto the
     pruner);
   - drive to a point behind the stance at the tool, look at the tool from there, move in, **take it**: the
     shears at their middle from straight above, the pruner by its handles with a slanted hand; hold it in front
     of the chest;
   - drive to a point behind the stance at the toolbox, look at the box, move in; one hand **takes the handle
     and leads the lid up** along its arc and holds it; the other **lets the tool go** over the open box; the
     first leads the lid back down and lets go: shut, once both tools are in; 0.2 rad from shut after the
     first, so that the lid comes down the rest of the way by itself.
   A tool that a hand did not get, or that is not where it was, is looked for again; a visit that did not put
   its tool in is made again from another stance.
4. **If the episode goes on after both tools were let go over the box**, one of them is not in it (the episode
   ends at once when the kit is together): the floor round the box is looked at from its other sides, and a tool
   that lies there is taken and put in again.

## Why it is built this way

- **One hand holds the lid while the other lets the tool go.** A lid that is let go does not stay half open, and
  one that is thrown open has its handle underneath and out of reach of a hand at the hinge side.
- **The handle is taken from above, the fingers closing across the bar, and only one way round.** The hand turned
  half a turn about its fingers takes the bar just the same, but then the lid stays shut and the whole box comes
  up with the hand (17 times of 18 over every side and either hand, against 18 of 18 the right way round).
- **The hand's middle goes 1.2 cm to the hinge side of the bar's middle.** The finger on the front side then
  brings the bar up against the other finger, which has closed all the way, and the lid follows the hand (142
  leads of 142 in trials and development runs). With the bar in the middle between the fingers both stop on it,
  and then the box came up with the hand, the hand stalled on the way or the lid was left open (6 times of 8).
  If both fingers have stopped on the bar, the hand lets go and takes it once more, 1 cm further to the hinge
  side.
- **The lid is led along an arc worked out from the hand itself**: from where the hand is once it has closed
  and where its fingers stopped, the bar's place is known to a millimetre or two, and the hinge line with it. Led
  along the arc planned from the look, the hand stalled on leads where the bar was 1 to 2 cm from where the plan
  put it, and let go there the lid fell open. A hand that stalls all the same (about one lead in nine) holds the
  lid where it is, which is open enough from 0.95 rad on; the way back starts from where the hand is; let go at
  the 1.2 rad where a way back sometimes stalls, the lid falls shut.
- **The hand shuts the lid itself, but only on the last tool.** Let go at 0.10 rad, the lid fell onto a pruner
  that stood high in the box and came to rest at 0.135 rad, just short of what counts as open, and with the
  shears in as well at 0.163. Shut by the hand after the first tool, it was once pressed onto shears that stood
  on edge against an end wall, and stuck: at the next visit the lid stayed shut and the whole box came up with
  the hand (start 315). After the first tool the hand therefore lets the lid go 0.2 rad from shut.
- **Where the way back of the lid stalls, the hand is opened and goes on along the arc.** On about one visit
  in nine (18 of 161) the hand stops a step from the top on its way back; why is not known, no contact of the arm
  shows there. Let go up there, the lid stays where it is or goes whichever way the hand's withdrawal sends it: 16
  times it fell shut, once open, and the box was left so (start 307). With the hand open the finger behind the
  bar takes the lid along, and the bar has room between the fingers: the lid came down shut on 3 such visits of 3
  (on a fourth the hand with the tool had thrown the lid open before, and there was no lid on the way).
- **After a visit that did not put its tool in, the box is looked for afresh**, not where it stood: the box
  that came up with the hand was set down 7 cm away, the look that followed kept it where it was known (to
  5 mm), and the hand closed three times on nothing.
- **Which hand opens the lid follows from the side the base stands at.** With the hand the right way round, the
  two-hand plan exists for the right hand from the hinge side and from the end at which the hinge is on its
  right, and for the left hand from the other end. The stance at the box is therefore chosen before the tool
  is taken, and the tool goes into the hand that is then free. Where either hand can open the box, the pruner
  goes to the hand that has the sound way onto it (from the end its blades lean away from, the base turned at
  most 0.6 rad from the hand): over the whole runs of five code states the other ways knocked it over on 5
  pinches of 13, the sound way on 1 of 39.
- **Where the tool is let go.** Over the open half of the box, 0.15 to 0.19 m along the box from the hand on
  the lid and never further than 0.20 m from the box's middle (its end walls are at 0.29), with what stands out
  of the hand pointing to the middle: the pruner stands 11 cm out of the hand that took it, 45 degrees off the
  fingers. Let go 0.24 m from the lid's hand it came down on the end wall and lay outside (twice, and the shears
  twice, once on the robot's own chassis); let go across the box under a lid led to 1.0 rad it went over the
  front wall. From an end of the box the far side of the lid's hand comes first.
- **The second tool goes into the other half of the box.** Let go onto the shears, the pruner threw them out
  of the box: they lay 0.4 m beyond its end; let go onto the pruner, the shears flew out as well (start 310). The
  same half is taken only where no stance at the box has a way to the other one.
- **A tool that came down outside the box is found by the look after the visit, or by a look round the box.**
  From the point behind the stance the camera shows the floor next to the box on its near sides: a tool that lies
  there is taken again (5 times in the development runs; a sixth time see below). The floor up to 0.4 m beyond the
  box it does not show: shears that came down on the front wall lay 10 cm in front of the box, unseen, and the
  run ended with the controller taking the kit for complete (start 312). The episode going on is the sign of
  that; the look round the box found the shears on 2 starts of 2, and they were put in.
  What a look shows within 5 cm of the box's own outline is the box, not a tool: seen from 0.85 m off its end,
  with that end at the image's lower border, a piece of the end wall was taken for the shears once, and the hand
  that went for them opened the lid a crack and turned the box (start 305). The five tools that did lie next to
  the box lay 9 cm or more from its outline.
- **The hand with the tool comes over the box 12 cm higher first, then straight down.** It crosses the lid's
  upper edge on its way. Coming straight from where it carried the tool it met the edge with its wrist camera and
  stopped there on about one visit in seven (the arm hangs 1 to 2 cm lower than planned when it reaches far);
  twice it shoved the lid and the box round with it. A place over the box where the hand has no posture that
  much higher is taken only if no other has one (2 of 4 drops from such places threw the lid open or shoved the
  box). The open box is also rounded more generously for that arm (spheres of 4 cm), with the plainer model as
  the second choice where that leaves no posture.
- **After a visit the base looks at the box from where it backed off to.** A tool that lies on the floor next
  to the box is taken again. For that look the toolbox, where it is known to stand, is set apart from what lies
  beside it (a pruner 5 cm from its wall was one thing with it).
- **The stance at the box is planned with the tool as the hand will hold it.** A plan made with empty hands was
  not there any more once the pruner was in the hand, at a box in a corner.
- **From the front where the other sides have no room.** A box that stands with its back to a wall (about 1
  development start in 20) leaves room for the chassis at its front only. From there the lid opens away from the
  robot, and the hand leads it to 1.05 rad only: led further, the arm that reaches over the box has no room
  between the lid and the wall. Both tools went in on two starts of two; with the lid at 0.8 rad four tools of
  nine fell outside. The same lower angle is the second choice from the other sides.
- **Pulling the box away from the wall by its handle does not work**: it tips over onto its front (9 of 9).
- **The pruner is taken by its handles with a slanted hand, along the pair, from the end its blades lean away
  from.** From straight above the palm meets the blades before the fingers are down at the handles, and the
  pruner falls (4 of 5). With the fingers pointing down by 45 degrees and coming along the pair of handles, they
  close across its 2.9 cm with 3 cm to spare on a side. From the end the blades lean away from, the pruner
  stayed upright in the hand (4 of 4 in trials; in whole runs 38 pinches of 39 held, with the base turned up to
  0.6 rad from the hand's way). From the other end it tilts in the hand or falls: held 7 times of 13 in a trial,
  and no better at a steeper hand (60 degrees), higher on the handles (10.5 cm) or further past their middle
  (3 cm); in whole runs 4 of 7. That end, and a base turned 0.9 rad from the hand's way (4 of 6), are used only
  where nothing else has a stance, next to a wall. Knocked over from that end, the pruner came to lie on a
  narrow side 5 times of 6, which is the way it is taken again most surely (below). Which end is which, the
  look at the pruner tells: the blades' top stands 2 cm beyond the middle of the pair, towards the end they lean
  to (right on 33 of 33 recorded looks).
- **The shears are taken at their middle, where the blades cross**, from straight above, the fingers closing
  across them 0.8 cm above the floor. There the fingers stayed 1.5 to 1.8 cm apart and the shears where they were
  in the hand through the lift, the way in front of the chest, the trunk's turns and a far reach (6 of 6 in a
  trial with that way). Taken 1.5 to 2.5 cm nearer the blades' points, as they were at first, they held at the
  lift (15 of 16) but turned out from between the fingers on the way and hung by the grasp alone, the fingers
  shut; the hand then lost them on about one visit to the box in ten. At their ring handles a finger goes
  through a ring and they hang on it without a grasp; 0.4 cm above the floor the fingers jam on it.
- **A pruner that has been knocked over is taken from the floor.** It comes to lie in one of two ways, and
  shows as such (129 of 142 recorded looks; never taken for the shears or for the pruner standing).
  *On a narrow side*, one handle on the other, 7 cm high and 3 cm across: so a hand that came along the pair of
  handles leaves it. A hand from straight above takes it across those 3 cm at the middle of its length, the
  fingers' ends 3 cm up, as it takes the pruner standing (6 of 6 in trials).
  *On a broad side*, 3 cm high and 6.3 cm across at its handles: so the chassis leaves it. The fingers open 10 cm:
  from above, one finger shoves the pair against the other, which stands at the end of its travel, and it is
  held by that squeeze alone (the simulator's assisted grasp takes hold only where the blades' joint has given
  and one handle is between the fingers). It is taken 2 cm from the middle of its length, next to where its
  weight is: there it hung nearly level and stayed in the hand through a shake 4 times of 4; taken 3 cm nearer
  the handles' ends it hung blades down and fell 3 times of 6. Across its blades, which are 0.7 cm thin, the hand
  came away empty. If the pruner is not in the hand at the box any more, it is looked for and taken again.
- **The chassis keeps clear of the tools**: at the stances at the box, at the spots it looks from and on the way.
  Before that it ran over the pruner on the way to the box, and turned on the shears.
- **Where the things are comes from the head camera's depth image**: points that stand above the floor where
  the map has free floor are grouped, and a group is one of the three things by its height and size. Colour is
  used for one thing, the latches. A thing that shows somewhere else than it was known is believed only if the
  place it was known at is in view and empty.
- **A tool that the hand closed on nothing at is forgotten** if a look does not show it there, or at once the
  second time: it has been pushed or has fallen, and is looked for again.
- **Whether a tool is still in the hand is read from the fingers**: the pruner keeps them 3 cm apart, the
  shears 1.5 cm. Fingers that have shut do not tell that the tool is gone: it may have turned out from between
  them and hang on by the simulator's grasp. Such a hand goes to the box all the same and is opened over it (a
  pruner taken from the end its blades lean to hung so over the open box; the hand was taken back shut, and
  opened later it let the pruner fall flat on the floor); the tool is then looked for, in case it had fallen
  before. Only the pruner taken lying flat, which no grasp holds, is looked for at once.
- **For the arm planner the shears in the hand reach 13 cm below the fingers.** Taken for a ball round the
  fingers' ends, and hanging from their blades, they were torn out of the hand on 2 of 17 visits from an end of
  the box, where the trunk bends furthest, and lay on the robot's own chassis.
- **A lid that the wall margin cuts is taken to its full length.** Points nearer than 10 cm to what the map has
  are not looked at, and a box that stands with an end towards a wall, as near as the starts have it, loses up to
  10 cm of that end of its lid. Its other end shows, and the lid is 0.61 m long. (3 of 280 development starts;
  on one of them the box was never seen whole and the look for its front never told.)
- **The localiser changes to another start pose only if that pose also explains the present view better.** The
  garage looks much the same from two opposite corners, and one start in nine begins with a second plausible
  pose that is followed along. As inherited, the localiser took the other pose once it had explained the views
  better for a while, also while the pose it followed matched the present view with 0.99: on start 309 the two
  changed places five times and the search ended 6 m off. The other poses are dropped once a tool has been taken
  where the map pose had it.
- **The door to the corridor is a wall for what stands on the floor.** The map keeps its leaf apart from what
  blocks the base; its foot was taken for the toolbox once, from 3.5 m. And a toolbox whose lid two looks from
  near do not show whole is not one: the search goes on without it.

## Limits known before the scored run

- A toolbox with a wall or the car close behind it AND no room to lead the lid even to 1.05 rad from the front
  (6 of 280 development starts), or with no stance at all (1): no visit is planned, the tools stay out.
- Both tools standing where the only stance at the box is: not handled (1 of 280).
- A tool that fell outside the box where the box hides it from the base's look is found only once both tools
  have been let go and the episode goes on (about 1500 to 2500 steps more); one that fell onto the robot itself
  is not found.
- A lid that has fallen open (it lies back, 2.97 rad) is not shut again. It fell so once in the development
  runs, after a way back that stalled; the hand now pushes the lid on in that case (3 times of 3).
- The shears let go over the box from a stance at its hinge side were outside it afterwards 7 times of 16 in
  the development runs of the last five code states (5 of them a metre and more away), from a stance at an end 2
  times of 24; the pruner was in the box after every one of 45 visits. Shears that lie on the floor are found
  and put in again; shears that came to rest on top of the lid (start 319) are not. A change that sends the
  shears to an end of the box where one has a stance was written but not checked before the freeze, and is
  not in this controller.
- A pruner that comes to stand in the box, or to hang on an end wall, holds the lid up, and the hand cannot
  shut it: let go 0.15 m from the lid's hand it ended 0.28 m along the box once, upright against the end wall,
  its middle 0.19 m above the floor, and the lid stayed at 0.5 rad (start 308). The fingers tell it (the bar
  turns between them when the lid does not come down: 3.2 cm apart, not 2.4), but nothing is done about it.
- From an end of the box the hand with the tool often has no posture 12 cm higher over the place where it lets
  go (23 visits of 83), and goes there straight. One such visit of 23 threw the lid open and tipped the box
  over (start 186, a development start).
- A pruner next to a wall may have a stance only for a hand that comes from the end its blades lean to, or
  with the base turned 0.9 rad from the hand: such pinches knocked it over 5 times of 13. It is then taken from
  the floor.
- A pruner taken from the floor is mostly held by the fingers' squeeze alone and can fall on the way. It is
  then taken again, which costs about 1500 steps a time.
- The front of the toolbox is told from the colour image. A look that is too far or shows the face only in
  part gives no verdict, and the look is repeated from another spot; a wrong verdict would send the hand onto
  the handle the wrong way round.

## Files

| File | What |
|---|---|
| `observation_policy.py` | Entry point for the native evaluator |
| `kit_observed.py` | What is known of the three things; the script's entry (`observed_pipeline`) |
| `kit_view.py` | The toolbox, the pruner and the shears from the head camera; the toolbox's front |
| `kit_pipeline.py` | The task as a sequence of skills: finding, choosing, taking a tool, the visit to the box |
| `kit.py` | The toolbox's geometry; the lid's handle and arc; the pinches; the two-hand plan of a visit |
| `car_observed.py`, `localizer.py`, `observer.py`, `scene_map.py`, `priors/house_map.npz` | Localisation and map (from c01 / u01) |
| `manip.py`, `wbik.py`, `skills.py`, `kinematics.py`, `robot_collision*`, `car_skills.py`, `car_pipeline.py` | Arm planner, base driving and the shared skills (from c01 / u01) |
| `run_native.py`, `evaluator_debug.py` | Native runner |
