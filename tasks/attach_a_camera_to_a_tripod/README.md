# attach_a_camera_to_a_tripod (2026 task 35)

Motion planner on RGB-D + proprioception (BIMBO_EvoX fair planner), served by `solutions/planner/backend.py`.

- Source: BIMBO_EvoX `0ed6e8236c`, `tasks/35_attach_a_camera_to_a_tripod/planners/fair`
- Package in this folder: `root/tasks/35_attach_a_camera_to_a_tripod/planners/fair` (plus the shared `core/skills_lib`, `skills/gpu-ik` and the IK worker of the same commit)
- Evolution: run `fair-35-r1`, program `d20842d2` (iteration 23), bundle sha256 `d4b2cae1ad46b9ea...`
- Measured (H100 port, seeds 301-310): SR 10/10, Q 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00
- Note: Evolution champion of fair-35-r1; seed fair planner W9 was SR 7/10 on the same seeds.

Regenerate with `tools/import_bimbo_planner.py` (the command is in the repository README).
