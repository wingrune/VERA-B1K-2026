# b01: scrubbing the bathroom floor, classical perception and planning

A classical controller for `scrubbing_bathroom_floor`, with no learned parts. The localisation, the map, the routes and
the whole-body arm planner come from the earlier controllers in this repository (`c01`/`u01`/`p01`). What is new is in
`scrub_view.py`, `scrub_observed.py`, `scrub_pipeline.py` and `heading_ways.py`.

The other files under `tasks/scrubbing_bathroom_floor/` (`code/`, `old_submissions/`, `results/`, `README.md`, ...)
belong to an earlier controller of the team and are not used by this one.

Development and every check ran only on the public starts 301-320.

## The task, as the simulator has it

- **The scene.** The scene is `house_single_floor`, bathroom floor x 19.73-24.99, y 9.71-11.97. A red scrub brush
  (15 × 7 × 3 cm) lies on the floor, with 1-7 dark marks of dirt. At the end no dirt may be left: one predicate, so
  Q is 0 or 1. Horizon 4730 steps.
- **How dirt goes.** A dirt particle goes when it lies within the brush's bounding box grown by 2 cm. The brush is held
  1.2 cm over the floor and moved over the mark.
- **Places the route planner cannot enter.** The route planner takes the base for a disc 0.40 m round, which does not
  fit into:
  - the nook by the toilet (0.93 m wide);
  - the strip between the bath and the basins' counter (0.77 m);
  - the shower.
  `heading_ways.py` plans slides at one fixed heading through them for the holonomic base.
- **Localisation traps.** The window behind the bath and the sliding door are glass, and the mirrors over the basins
  and the toilet reflect the room in depth. Those regions are left out of the scan fit (`ScrubEstimator`).
- **Where dirt can be.** Dirt is put only along fixed bands of the floor (`priors/dirt_band.npz`): marks outside them
  are false, and coverage is counted over them only.
- **What the map's heights cannot express.** They are column tops: the chassis fits under the counter, but for the arm
  planner it then stands inside it. Stances there are not used (`under_nothing`).

## What it does

1. **Look for the brush:** on the spot, then from lookouts over floor not yet seen.
2. **Take the brush** from above with the hand squatting beside the base, fingers across its width, at the cheaper
   stance of either hand. Before closing, a look with the wrist camera.
3. **Survey** the floor from the places that show most of what is still unseen. It stops when under 3 % is unseen or
   after 450 steps.
4. **Look into the nook by the toilet** from its mouth, facing north, if its far end is unseen and the run is still
   under 2000 steps.
5. **Scrub each mark.**
   - **Order:** marks never scrubbed first, the nearest of them. The marks in the strip by the bath are taken
     together, once the base is in it or the rest are done.
   - **Stance:** in the strip, a stance at the heading the base already has is slid to directly, without going out and
     in again.
   - **The scrub:** the brush down 1.2 cm over the floor, moved 5 cm to and fro, the arm folded for the way on. Other
     marks under the swept box count as done.
6. **Look round and repeat** while marks are left and there is time. When none is left: a look at the strip's east end
   from inside the strip, then a further survey.

## Known limits (dev rounds on 301-320)

- **Time.** A run with 5-7 marks, or marks in the strip by the bath, does not finish in 4730 steps. A mark costs about
  200-350 steps; a visit to the strip 700-1250.
- **Starts 307 and 316:** the brush lies at the east end of the strip, where the map's column heights leave no stance
  for the robot. The brush is never taken.
- **Hidden floor.** Floor under the robot's starting place, and in the nook's far corner, is sometimes never seen.
