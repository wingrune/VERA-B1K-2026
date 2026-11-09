# hanging_pictures (2026 task 34)

RGB-D planner from the B1K_hacker evolution passport, served by `solutions/planner/backend.py`.

- Source file: `projects/B1K_hacker/h100_render/tasks/34_hanging_pictures/planners/rtx_v1/versions/evo_evo-20261008-165320_b07a9f5b.py`
- Copied byte-for-byte to `root/tasks/34_hanging_pictures/planners/rtx_v1/planner.py`
- Evolution: run `evo-20261008-165320`, program `b07a9f5b-0534-47db-8481-7a6b1cc231a4` (iteration 0, generation 0), sha256 `e922081805d696c74404af3e8d36ee01f1cebe7833b0f3b8b14da4d75fd6498a`
- Recorded measurement (native RTX, instances 0-9 / seeds 301-310, horizon 20000): SR 1/10, mean Q 0.1. Per-episode Q: 301=1.00, 302=0.00, 303=0.00, 304=0.00, 305=0.00, 306=0.00, 307=0.00, 308=0.00, 309=0.00, 310=0.00. This is the highest valid evolution Q; the same program is the only one at that Q. Later rtx_v1 runs top out at Q 0.
- The file imports only the standard library, NumPy, and SciPy. It does not import sibling modules, and it does not import cuRobo. Kinematics constants are inlined; nothing is loaded from outside this task directory.

This is the preserved evolution artifact. The official 301-310 RTX result is measured separately by `scripts/run_eval.sh`.
- RTX 4090, native evaluator, instances 301-310: SR 0/10, mean Q 0.00. Per-instance Q: 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00. HF commit `162b355aaccc3eab82e0ca6fc8a7e56a1ff5bac5`.
