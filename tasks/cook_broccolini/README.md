# cook_broccolini (2026 task 96)

RGB-D planner from the B1K_hacker evolution passport, served by `solutions/planner/backend.py`.

- Source file: `projects/B1K_hacker/h100_render/tasks/96_cook_broccolini/planners/broc_v2/versions/evo_evo-20261010-131913_cb2a002c.py`
- Copied byte-for-byte to `root/tasks/96_cook_broccolini/planners/broc_v2/planner.py`
- Evolution: run `evo-20261010-131913`, program `cb2a002c-4966-499a-a835-b4487b96d39b` (iteration 3), sha256 `b4d974bcfef0271cf37b13ca964d6c5d65607508e26a5d90a10d6bb9549f9427`
- Recorded measurement (H100 SR008, Isaac Lab port of BEHAVIOR, instances 301-310): SR 0/10, passport Q 0.178782 (evolution combined_score; unique highest among 11 valid programs). Episode mean Q 0.145455. Per-episode Q on 301-310: 0.273, 0.091, 0.000, 0.273, 0.273, 0.000, 0.000, 0.000, 0.273, 0.273. All ten episodes timed out at horizon 6927. The passport has no native RTX 301-310 measurement.
- Runtime imports are the standard library (`heapq`, `math`), NumPy, and SciPy. Those packages are already installed in the planner Docker venv (`numpy==2.4.6`, `scipy==1.17.1`).

This is the preserved evolution artifact. The official 301-310 RTX result is measured separately by `scripts/run_eval.sh`.
- RTX 4090, native evaluator, instances 301-310: SR 0/10, mean Q 0.00. Per-instance Q: 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00. HF commit `ff683a7f8053481f30aa43011835224097091a6a`.
