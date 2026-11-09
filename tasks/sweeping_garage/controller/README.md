# s03: sweeping the garage with a map prior and classical motion planning

Third controller for this task: `s02` (scored 8/20, mean Q 0.55, on the second twenty held-out starts) with the
broom's pose seen more truly and the broom knocked less (below, "What s03 changes"), which is `s01` (scored 6/20 on
the first twenty) with the broom taken more surely ("What s02 changes"). `s01` started from the controller for `put_together_a_basic_pruning_kit` (`p01`) and,
through it, from `c01`/`u01`: the same garage, localisation, routes and arm planner; the broom is found as `p01`
found its tools. What is new is taking a 1.36 m broom off the floor, holding it flat a centimetre and a half over the
floor and driving it over the whole floor by a fixed plan of straight sweeps.

## The task

The robot starts somewhere in the garage. A broom and a bucket lie on the floor; about 25 particles of sand and dust
lie on the floor round the parked car. At the end no particle of either may be left (the floor not `Covered` by
either). Horizon 6709 steps (1.5 x the human mean of 4473).

What the simulator makes of this, measured (`dev/jobs/probe_static.py`, `probe_sweep.py`, `broom_trial.py`) or read in
its source:

- A particle goes when it is inside the broom's bounding box (axis-aligned in the scene) with a 2 cm margin
  (`ParticleRemover`): the broom need only pass over it, its lowest point under about 2.5 cm. A broom tilted by a few
  degrees in the hand lifts its head 5 cm and misses particles.
- The broom is one rigid body: a head 0.45 m wide, 0.30 deep and 0.16 high (bristles), a thin handle from the head's
  top down to the floor at its end, 1.21 m from the head's middle. The hand holds it by assisted grasp (a fixed joint).
- Garage floor x -9.27 to -1.47, y 2.26 to 8.75; the car stands on x -6.85 to -1.78, y 5.82 to 7.97. The particles
  lie in an L round it: the strip by the garage door (x -9.27 to -6.85, the garage's whole depth) and the band in
  front of the car (y 2.75 to 5.5). The garage door is shut at x -8.84 to -8.93: in 22 of the 280 training starts
  particles lie behind it, out of reach.

## What it does

1. **Localise** in the garage and **find the broom** from the head camera's depth image: a group of points above the
   free floor 0.45-1.6 m long and under 0.21 m high (head and handle seen as two groups are taken together); its head
   is the end where the points stand highest.
2. **Take it in the side hold**: the base stands beyond the handle's end, its heading across the handle; the hand
   takes the handle 1.08 m from the head's middle, beside the base's middle and 0.55 m out to the hand's side, the
   trunk squatting over the base (a condition on the whole-body IK keeps the trunk within 0.45 m ahead and 0.60 m
   behind the base's middle), and lifts the broom 1.5 cm. The right hand when the base has room for it so, else the
   left. (Held 0.5 m ahead of the base with the trunk bent over it, the head camera stood 1.0 m ahead and met the
   walls at the ends of the sweeps, which shook the broom off the floor.)
3. **Where the base has no room beyond the handle's end** (a wall or the car): take the broom in front of the base
   (0.5 m ahead, at its end or nearer the head; or along it, from beyond its end, where it lies between the car and
   the garage door), back away the way the base came, carry it keeping the heading to the nearest spot where, put
   down, it leaves room for the side hold even a few centimetres or degrees off, put it down there and take it again
   (twice at most).
4. **Sweep** the plan for that hand (`sweep.py` `PLANS`: eight straight legs, 34.7 m, found offline over the training
   starts so that neighbouring legs overlap by a few centimetres): each leg driven either way along its heading at
   0.45 m/s. The order is planned again from where the base is after each leg, for the least time (a dynamic
   programme over move times between the legs' ends computed offline, `priors/sweep_moves.json`); between legs the
   base drives, keeping its heading, to a spot where it can turn with the broom held out and turns in steps of
   1.5 rad. The legs are shifted by where the hand actually took the handle. While the broom is held the localiser
   keeps correcting the pose with the broom cut out of the scans (otherwise the pose drifted 0.6-0.9 m in one sweep).

## What s02 changes

`s01` lost most starts before the first sweep: the base had no stance by the handle's end, the fingers closed on
nothing, or the broom was knocked round on the way in. In development on the public starts it solved 8-9 of 20,
`s02` 13 of 20 (`dev_runs_public.txt` in the scored run's folder).

- **The handle's height** is taken from the broom's mesh: the handle, 3.4 cm thick, leaves the head's top 0.125 m up
  (its middle) and meets the floor 1.28 m from the head's middle (`sweep.handle_z`); the straight line of `s01` put
  the fingers 3 cm over the handle near the head.
- **A last look of the wrist camera** with the hand 10 cm over the handle (`sweep_observed.wrist_handle`): the
  handle's place across it and its height over the floor the camera sees; the pinch is planned again there.
- **The broom is a thing lying about for the arm planner** while the hand goes in (its handle and head as balls):
  on its way to the start of the pinch the arm had swept the handle round.
- **Pulled out first** where no hold has room by the handle's end (it lies under the car's side): the handle is
  taken near the head and the base slides back the handle's way by 0.8 m (or 0.5), twice at most; then the usual
  holds.
- **Looked at again after a put-down** from 0.6 m back, where the whole broom is in view (the hand's place had put
  it 7 cm off), and the 'along' hold takes the handle 1.05 m from the head (the chassis' front 10 cm off its end).

## What s03 changes

`s02` lost 4 of its 20 scored starts with the broom not taken and, in development, three public starts (302, 312,
317) the same way: the broom knocked by the base on its way to a stance, or after a put-down seen 15-26 degrees off,
and the hand then went 0.3 m wide of the handle. In development on the public starts `s03` reached a mean Q of
about 0.85 (`dev_runs_public.txt` in the scored run's folder).

- **The broom's line from its handle** (`broom_view.py`): the main direction of all of the broom's points leans
  towards the head, 0.44 m across the handle; the handle's own points, 0.3 m and more from the head's end, give its
  line, the head's points its middle. Seen from its side after a put-down it was 15-26 degrees off before, 8 after.
- **No pinch in the air**: where the wrist camera shows no handle under the hand, the base backs off 0.6 m and looks
  again, now turned to where the broom was last seen and to either side of that (it had been knocked beside the
  base, out of the look straight ahead).
- **Stances on the robot's side of the broom first**: the way round the handle's end to a stance on the far side
  knocked the broom where there was no room to keep away from it.
- **Pulled out** by the handle near the head when no hold from in front has room either (after a knock), then the
  usual holds.
- **Routes round the lying broom** keep as much room from it as there is (0.35 m down to none), instead of all or
  nothing.

## Files

`broom_view.py` (detector), `sweep_observed.py` (estimator and the pipeline's entry), `sweep.py` (holds, the broom's
boxes, the plans), `sweep_pipeline.py` (the task), `observation_policy.py` (entry for the native evaluator via
`run_native.py`); the rest is shared with `p01`/`c01`/`u01` (localiser, observer, map, routes, whole-body IK, arm
planner). Offline tools in `dev/`: `plan_sweep.py` (the plans), `sweep_moves.py` (the move times), `study_sweep.py`
(a plan's coverage over the starts).
