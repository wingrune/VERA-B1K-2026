# chop_an_onion (2026 task 42)

Motion planner on RGB-D + proprioception (BIMBO_EvoX fair planner), served by `solutions/planner/backend.py`.

- Source: BIMBO_EvoX `106e2be531`, `tasks/42_chop_an_onion/planners/fair`
- Package in this folder: `root/tasks/42_chop_an_onion/planners/fair` (plus the shared `core/skills_lib`, `skills/gpu-ik` and the IK worker of the same commit)
- Evolution: run `fair-42-r3`, program `00403145` (iteration 143), bundle sha256 `65a4628605f0343f...`
- Measured (H100 port, seeds 301-310): SR 1/10, Q 0.00 0.75 0.00 0.75 0.00 0.00 1.00 0.00 0.75 0.00
- RTX 4090, native evaluator, image `b1k-evo-submit:tri-29664e4`, instances 301-310: SR 0/10, mean Q 0.15. Per-instance Q: 301=0.00, 302=0.25, 303=0.00, 304=0.75, 305=0.00, 306=0.00, 307=0.50, 308=0.00, 309=0.00, 310=0.00. All ten stopped at the 9600-step horizon. All 10 JSON and 10 MP4 passed `scripts/check_results.py --metrics-only --strict`. [Artifacts](https://huggingface.co/datasets/VERA-VLA/b1k_final_submission/tree/4ccb33cb0f27092e29b3cbb550bdfe69035690f4/standard.public.VERA_team.team.20261015) (HF commit `4ccb33cb0f27092e29b3cbb550bdfe69035690f4`).
- Note: Best of fair-42-r3, H100 fitness 0.3025. `phases/board_hunt.py` imports `fair_toolbox` from the same pin; that package is at `root/skills/fair-toolbox/fair_toolbox` and linked from `root/fair_toolbox`, so its cuRobo helper resolves to `root/skills/gpu-ik`.

Regenerate with `tools/import_bimbo_planner.py` (the command is in the repository README).
