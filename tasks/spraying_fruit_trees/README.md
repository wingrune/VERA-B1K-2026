# spraying_fruit_trees (2026 task 39)

RGB-D planner from the B1K_hacker evolution passport, served by `solutions/planner/backend.py`.

- Source file: `projects/B1K_hacker/h100_render/tasks/39_spraying_fruit_trees/planners/spraytrees_v1/versions/evo_evo-20261010-132017_0698e8f4.py`
- Copied byte-for-byte to `root/tasks/39_spraying_fruit_trees/planners/spraytrees_v1/planner.py`
- Evolution: run `evo-20261010-132017`, program `0698e8f4-9e95-4a31-ae5f-500d28d14fb3` (iteration 1), sha256 `f421be54ea8633cbfa3e92de01423ce6270272bf6e9412f2852bab3d61324baa`
- Recorded measurement (H100 port, instances 0-4 / native seeds 301-305): SR 0/5, mean Q 0.10. The passport page labels combined_score 0.1198 as Q. Per-episode Q: 301=0.50, 302=0.00, 303=0.00, 304=0.00, 305=0.00. Nineteen of 50 valid programs share mean Q 0.10 and SR 0; this is the latest. The passport has no separate native RTX 301-310 run.
- Runtime dependencies are NumPy and SciPy, already installed in the planner Docker venv. The file does not import torch, OpenCV, scikit-image, Warp, or cuRobo, and it does not load sibling modules.

This is the preserved evolution artifact. The official 301-310 RTX result is measured separately by `scripts/run_eval.sh`.
- RTX 4090, native evaluator, instances 301-310: SR 0/10, mean Q 0.00. Per-instance Q: 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00. HF commit `894977f02983a4bde965acd679db09477772c4a2`.
