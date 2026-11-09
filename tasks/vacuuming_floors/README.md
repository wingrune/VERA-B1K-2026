# vacuuming_floors (2026 task 69)

Motion planner on RGB-D + proprioception (BIMBO_EvoX fair planner), served by `solutions/planner/backend.py`.

- Source: BIMBO_EvoX `106e2be531`, `tasks/69_vacuuming_floors/planners/fair`
- Package in this folder: `root/tasks/69_vacuuming_floors/planners/fair` (plus the shared `core/skills_lib`, `skills/gpu-ik` and the IK worker of the same commit)
- Evolution: run `fair-69-r1`, program `434d955b` (iteration 68), bundle sha256 `502bd01ed73a4af5...`
- Measured (H100 port, seeds 301-310): SR 3/10, Q 0.00 0.00 1.00 0.00 0.00 1.00 0.00 0.00 0.00 1.00
- RTX 4090, native evaluator, image `b1k-evo-submit:vacuum-434d955b` (`sha256:e970c08e210fe07ee2f71728178845d8df821cef78b739ef26b0ec8ec7daae14`), instances 301-310: SR 0/10, mean Q 0.00. Per-instance Q: 301=0.00, 302=0.00, 303=0.00, 304=0.00, 305=0.00, 306=0.00, 307=0.00, 308=0.00, 309=0.00, 310=0.00. Every episode stopped at 3618 steps. All 10 JSON and 10 MP4 passed `scripts/check_results.py --metrics-only --strict`, and their SHA256 and Hugging Face object hashes were checked. [Artifacts](https://huggingface.co/datasets/VERA-VLA/b1k_final_submission/tree/b0e59a14be9e6a39731c41a75b7f747d50872085/standard.public.VERA_team.team.20261015) (HF commit `b0e59a14be9e6a39731c41a75b7f747d50872085`).
- Note: Best of fair-69-r1, H100 fitness 0.3405

Regenerate with `tools/import_bimbo_planner.py` (the command is in the repository README).
