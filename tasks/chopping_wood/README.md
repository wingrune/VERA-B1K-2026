# chopping_wood (2026 task 44)

RGB-D planner from the B1K_hacker evolution passport, served by `solutions/planner/backend.py`.

- Source file: `projects/B1K_hacker/h100_render/tasks/44_chopping_wood/planners/rtx_v3/versions/evo_evo-20261010-132057_b6615d11.py`
- Copied byte-for-byte to `root/tasks/44_chopping_wood/planners/rtx_v3/planner.py`
- Evolution: run `evo-20261010-132057`, program `b6615d11-c5b7-4afd-9d2c-5275256e2793` (iteration 0), sha256 `359e5a5c413343c8f6a3374fb03cd9e82d267ad1db7a30eff8340337bd594c7b`
- Recorded measurement (native RTX, instances 301-305 only): SR 3/5, mean Q 0.75. Per-episode Q: 301=1.00, 302=0.00, 303=1.00, 304=1.00, 305=0.75. The passport has no separate H100 301-310 run.
- Runtime dependencies are NumPy and SciPy, already installed in the planner Docker venv. The file does not import cuRobo.

This is the preserved evolution artifact. The official 301-310 RTX result is measured separately by `scripts/run_eval.sh`.
- RTX 4090, native evaluator, instances 301-310: SR 4/10, mean Q 0.60. Per-instance Q: 1.00, 0.00, 1.00, 0.00, 0.75, 0.50, 1.00, 1.00, 0.75, 0.00. HF commit `f436dab73e5e60f87218425f364527ca9528a21f`.
