# installing_a_scanner (2026 task 92)

Motion planner on RGB-D + proprioception (BIMBO_EvoX fair planner), served by `solutions/planner/backend.py`.

- Source: BIMBO_EvoX `32da8e4035`, `tasks/98_installing_scanner/planners/fair_ag`
- Package in this folder: `root/tasks/98_installing_scanner/planners/fair_ag` (plus the shared `core/skills_lib`, `skills/gpu-ik` and the IK worker of the same commit)
- Evolution: run `fair-98-r3`, program `7dd85299` (iteration 55), bundle sha256 `d2174ab2fa78eeb9...`
- Measured (H100 port, seeds 301-310): SR 4/10, Q 1.00 0.50 1.00 0.50 0.50 0.00 1.00 0.50 1.00 0.00
- Note: Evolution champion of fair-98-r3; the seed fair f5 scored native RTX 1.0 on 301.

Regenerate with `tools/import_bimbo_planner.py` (the command is in the repository README).
- RTX 4090, native evaluator, instances 301-310: SR 3/10, mean Q 0.40. Per-instance Q: 0.00, 0.50, 1.00, 0.00, 0.00, 0.00, 1.00, 0.50, 1.00, 0.00. HF commit `5f6b975fe8c9b7757ce858b03529774e17d7bf34`.
