# h01: putting away the Halloween decorations with a map prior and classical motion planning

First controller for this task. It starts from the controller for `put_together_a_basic_pruning_kit` (`p01`) and,
through it, from `c01`/`u01`: the same localisation, routes and arm planner. What is new is finding six things on the
living-room floor, taking a candle by its wick and a pumpkin by its stem, pulling the cabinet's drawers out by their
lip and pushing them shut, and setting the caldron down by the coffee table.

## The task

The robot starts somewhere in the living room. Two pumpkins, three pillar candles and a caldron stand on the floor,
anywhere on it. At the end the pumpkins and the candles must be inside the cabinet against the room's east wall, the
cabinet not open, and the caldron next to the coffee table. Horizon 20682 steps (1.5 x the human mean of 13788).

What the simulator makes of this, measured (`dev/jobs/probe_static.py`, `probe_decor.py`, `decor_trial.py`) or read
in its source:

- The cabinet `bottom_cabinet_rhdbzv_0` is 0.41 m high, its front at map x 6.015. Two drawers side by side
  (`j_left_drawer` y 4.59-5.65, `j_right_drawer` y 3.33-4.57) slide out along -x by up to 0.297 m and stay where
  they are let go. The cabinet counts as open when a drawer is out by 1.55 cm or more (5 % of its travel). Along the
  top of each drawer's front runs a lip 2 cm deep and 8 mm thick, 0.375 m up: the only handle.
- A thing is inside when the middle of its bounding box is inside the hull of a drawer's container volume. All five
  fit in one drawer and stay inside when it is shut.
- Pumpkin: 15 cm across (wider than the gripper's 10 cm), 12.6 cm high; a stem 2.8 x 1.8 cm up to 12.5 cm, leaning
  up to 1.2 cm from the middle. Pillar candle: 10.1 cm across, 11.7 cm high, a 5 mm wick up to 14 cm. Caldron:
  29.4 cm across, 25.2 cm high, open at the top.
- The caldron is next to the table when the gap between their bounding boxes is under about 18 cm (and a level ray
  from one meets the other); the table never moves (map 3.11-3.93 x 4.46-6.13).
- A pinch holds (assisted grasp) only when both fingers touch and a ray between them meets the thing: on a 5 mm wick
  the hand has about 1 mm to spare across its fingers.

## What it does

1. **Localise** in the living room and **find the six things** from the head camera's depth image (groups of points
   above free floor, told apart by height, size and colour: orange is a pumpkin): a look round on the spot, then from
   up to five lookouts.
2. **The caldron first if it stands where a drawer slides out** or where the base stands to pull one: it is taken by
   its rim and set down next to the table.
3. For each pumpkin and candle, the ones in a drawer's way first, then the nearest: drive to a point behind the
   stance, look at it from there, move in; the hand's position is carried from that look by dead reckoning (the map
   pose is put right by 2-3 cm on the way in), and the **wrist camera's last look** from above puts the hand right
   before it goes down. A candle is taken by its wick, a pumpkin by its stem, from straight above; lifted 0.40 m
   straight up and carried slowly in front of the chest.
4. **At the cabinet**: if the drawer of the next place is shut, the free hand takes its lip (a wrist-camera look at
   the drawer's front sets the depth), pulls it out 0.287 m and lets go; the thing is let go 6 cm above that drawer's
   floor over its place (three places in the left drawer, two in the right). A thing that cannot go in (no stance
   before either drawer) is set down out of the way, and the caldron is tried again.
5. **The drawers are pushed shut** with the closed hand while the base drives in to 5 cm past where the shut front is.
6. **The caldron** is set down next to the table, if it was not moved first; it is looked for again if the hand closed
   on nothing (the chassis can push it).

A thing that falls on the way is looked for afresh; a pinch that closed on nothing is tried from another stance.

## Files

`decor.py` (things, cabinet, pinches, drawer moves), `decor_view.py` (detector), `decor_observed.py` (estimator: what is
where, dead reckoning, wrist-camera looks), `decor_pipeline.py` (the task), `observation_policy.py` (entry for the
native evaluator via `run_native.py`); the rest is shared with `p01`/`c01`/`u01` (localiser, observer, map, routes,
whole-body IK, arm planner). `priors/house_map.npz` is built from this task's own scene probe.
