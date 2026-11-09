# t01: cleaning the trumpet with a tilted brush, classical perception and planning

A classical controller for `clean_a_trumpet` (task 37), with no learned parts. The localisation, the map, the routes and
the whole-body arm planner come from the earlier controllers in this repository (`c01`/`u01`/`p01`, see `manip.py`,
`car_*`, `localizer.py`, `scene_map.py`). What is new is in `trumpet_view.py`, `trumpet_observed.py` and
`trumpet_pipeline.py`.

Development and every check ran only on the public starts 301-320.

## The task, as the simulator has it

- **The scene.** The scene is `house_double_floor_upper`. A trumpet (`trumpet_87`) and a scrub brush (`scrub_brush_86`)
  lie on a desk (`desk_aduafr_0`, top at z 0.773) in a bedroom. About 20 visual dust particles lie on the trumpet's
  upper surface.
- **The goal.** The trumpet must not be `Covered` by dust at the end. Even one particle left fails the episode, and
  the predicate is all or nothing (Q is 0 or 1).
- **How a particle goes** (OmniGibson `ParticleRemover`, adjacency mode). It is removed, every step, when it lies in
  the brush link's axis-aligned bounding box grown by 2 cm. No contact is needed.
  - The brush need not touch the trumpet: it has to be held so that this box takes the particle.
  - The box is axis-aligned and the brush is held tilted, so the box is larger than the brush.
- **Where the dust lies.** The particles lie on the trumpet's collision surface: 3 mm under its top on median, over 595
  particles of the training files. The visual surface is 1-1.5 cm lower.
- **The stool.** A stool stands in front of the desk's middle (x -5.46 to -4.99). In the map, the room round its edge
  falls from +6 cm to -9 cm within 5 mm. Base stances there stand at y >= 0.8, with a 2 cm margin every way.

## What it does

1. **Localise; find the trumpet and the brush** from up to ten vantages round the desk, nearest first.
   - **The trumpet's pose:** a closest-point fit of its model to the head camera's yellow points (`trumpet_view.py`).
     The fit is planar, with z fixed by the desk, and starts from the last pose. A 3-D refinement runs only for poor
     planar fits.
   - **The brush:** its red points on the desk.
   - **A brush hidden from every vantage:** a look with the wrist camera, the hand held high over the trumpet's near
     side and turned to look past it (`search_behind`).
2. **Take the brush from above, 2.5 cm from its middle, by the end nearer the base.**
   - **Stance:** chosen by reach. The arm is planned with the trumpet as an obstacle; on the way over to the pinch's
     start not even the hand may touch it.
   - **Before closing:** a look with the wrist camera.
   - **After closing, the brush's place in the hand is corrected:**
     - by half the fingers' difference along their closing;
     - by 1.3 cm along the brush and 0.7 cm deeper (measured against the simulator's pose with the true base pose);
     - across, by the holding hand's own wrist camera, once the brush has been turned over the desk's near edge.
3. **Dab the dust.** The brush is held tilted 0.4-0.8 rad from upright, its lower end let down beside each target.
   - **Clearance:** the whole brush stays 6 mm over the trumpet's collision height field (`priors/trumpet_fields.npz`),
     grown by 6 mm.
   - **Attitudes:** thirteen, as (tilt, turn of the heading): the first that takes a target is used.
     - Nine with turns up to ±0.6.
     - Turns of ±1.2 and ±1.8, for the particles low in the bell's mouth.
   - **Targets:**
     - first the specks the head camera shows: dark greyish blobs on the trumpet, fused in its frame;
     - after two rounds with no more than two specks, or from step 4000, blind dabs as well: the surface points not yet
       passed over (every 1 cm, on the collision surface), likeliest dust first (`priors/dust_prior.npz`).
   - **Gliding:** between dabs closer than 0.35 m with the same attitude, the brush glides low (6 mm higher than a
     dab's bottom) instead of up, across and down.
   - **Collision checks on the way:** moves over the trumpet are checked against it. Every 4 dabs a look checks
     whether the trumpet has been pushed (3 cm or 0.09 rad), and if so the round is planned again.

## Known limits (dev rounds on 301-320)

- **Time.** About 10 of 20 runs end at the horizon (7960 steps) with 1-5 particles left. A run rarely finishes one
  blind pass of the surface.
  - Gliding stops at every centimetre: about 2.2 steps a centimetre, 3900 steps of a run.
  - Each look round costs about 130 steps.
- **The trumpet is pushed or tipped now and then**, by the arm, by the head on far dabs, or by the brush right after
  the lift. Its planar pose estimate then misses by centimetres, and in some runs it falls off the desk.
- **Start 317:** the brush lies 0.95 m from any stance, behind the trumpet at the desk's west end, and is never taken.

Work after this version (not in it), with the evidence, is in `../../../REPORT_20261015_T01.md`.
