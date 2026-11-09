# store_honey (2026 task 83)

Motion planner on RGB-D + proprioception (BIMBO_EvoX fair planner), served by `solutions/planner/backend.py`.

- Source: BIMBO_EvoX `106e2be531`, `tasks/83_store_honey/planners/fair_ag`
- Package in this folder: `root/tasks/83_store_honey/planners/fair_ag` (plus the shared `core/skills_lib`, `skills/gpu-ik` and the IK worker of the same commit)
- Evolution: run `fair-83-r1`, program `ffacbbe6` (iteration 201), bundle sha256 `fbf438cf79e3af73...`
- Measured (H100 port, seeds 301-310): SR 7/10, Q 1.00 1.00 1.00 0.00 1.00 1.00 0.00 0.00 1.00 1.00
- RTX 4090, native evaluator, image `b1k-evo-submit:store-honey-ffacbbe6` (`sha256:3445e5d4b5685ef53ac24b2a771140e0d1c42687f3097d33096d112f4258d9fd`), instances 301-310: SR 3/10, mean Q 0.30. Per-instance Q: 301=1.00, 302=0.00, 303=0.00, 304=0.00, 305=0.00, 306=0.00, 307=0.00, 308=1.00, 309=0.00, 310=1.00. Successes finished at 5265, 4515 and 6243 steps; the other seven stopped at the 10151-step horizon. All 10 JSON and 10 MP4 passed `scripts/check_results.py --metrics-only --strict`, and their SHA256 and Hugging Face object hashes were checked. [Artifacts](https://huggingface.co/datasets/VERA-VLA/b1k_final_submission/tree/82301cba5c2abd44393aac7cfacb624eb2a6a06f/standard.public.VERA_team.team.20261015) (HF commit `82301cba5c2abd44393aac7cfacb624eb2a6a06f`).

Regenerate with `tools/import_bimbo_planner.py` (the command is in the repository README).
