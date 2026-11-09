# boxing_food_after_dinner (2026 task 67)

RGB-D planner from the B1K_hacker evolution passport, served by `solutions/planner/backend.py`.

- Source file: `projects/B1K_hacker/h100_render/tasks/67_boxing_food_after_dinner/planners/boxing_v1/versions/evo_evo-20261009-163744_2bde53a4.py`
- Copied byte-for-byte to `root/tasks/67_boxing_food_after_dinner/planners/boxing_v1/planner.py`
- Evolution: run `evo-20261009-163744`, program `2bde53a4-3964-47a3-8130-1e94ff2a24c1` (iteration 0), sha256 `20b102469f02566e8f4ee3d377d5b4e80b560ac5a1889666b06b426c780ca0f6`
- Recorded measurement (H100 SR008, Isaac Lab port, instances 301-310): SR 0/10, mean Q 0.092593. Per-episode Q: 301=0.167, 302=0.000, 303=0.000, 304=0.000 (infrastructure), 305=0.000, 306=0.167, 307=0.167, 308=0.167, 309=0.167, 310=0.000. One of ten episodes failed on infrastructure; Q and success are over the nine valid episodes. The passport has no separate native RTX 301-310 run.
- Runtime dependencies are NumPy and SciPy, already installed in the planner Docker venv. The file does not import cuRobo. It imports no sibling modules.

This is the preserved evolution artifact. The official 301-310 RTX result is measured separately by `scripts/run_eval.sh`.
- RTX 4090, native evaluator, instances 301-310: SR 0/10, mean Q 0.05. Per-instance Q: 0.17, 0.00, 0.00, 0.17, 0.00, 0.00, 0.00, 0.17, 0.00, 0.00. HF commit `9c81e6e2e8f06c4359dd3f808e5ba9437e29611f`.
