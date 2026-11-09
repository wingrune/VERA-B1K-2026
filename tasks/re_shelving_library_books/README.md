# re_shelving_library_books (2026 task 55)

Motion planner on RGB-D + proprioception (BIMBO_EvoX fair planner), served by `solutions/planner/backend.py`.

- Source: BIMBO_EvoX `106e2be531`, `tasks/55_re_shelving_library_books/planners/fair_ag`
- Package in this folder: `root/tasks/55_re_shelving_library_books/planners/fair_ag` (plus the shared `core/skills_lib`, `skills/gpu-ik` and the IK worker of the same commit)
- Evolution: run `fair-55-r1`, program `c243b165` (iteration 200), bundle sha256 `7f75e2fd5ded0df9...`
- Measured (H100 port, seeds 301-310): SR 0/10, Q 0.33 0.33 0.33 0.33 0.00 0.33 0.33 0.33 0.33 0.67
- Note: Best of fair-55-r1, H100 fitness 0.3348

Regenerate with `tools/import_bimbo_planner.py` (the command is in the repository README).
- RTX 4090, native evaluator, instances 301-310: SR 0/10, mean Q 0.07. Per-instance Q: 0.00, 0.00, 0.00, 0.33, 0.00, 0.33, 0.00, 0.00, 0.00, 0.00. HF commit `9aa5d0fa81ed7d7f56b42f8dadc4af3f2f991e8f`.
