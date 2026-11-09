# polishing_shoes (2026 task 79)

Classical controller on RGB-D + proprioception (no learned parts) from the b1k_rl_guys repository, served by `solutions/b1k_classical/backend.py`.

- Source: `tasks/polishing_shoes/experiments/candidate/h02` at `54d6cd5fac` (branch `pick-and-place`), source fingerprint `8852ee4b200207b3…` (20 files), in `controller/`
- Inputs: head and wrist RGB-D, proprioception, robot-relative camera poses; static priors in `controller/priors/` (robot model, a map of the task's scene, object models); no instance pose and no simulator state
- Design and task facts: `controller/README.md`; scored runs and reports: b1k_rl_guys `tasks/polishing_shoes/`
- Measured (native evaluator, fresh process per episode, instances 301-310): SR 0/10, mean Q 0.27; per instance 301: q 0.00, 302: q 0.00, 303: q 0.33, 304: q 0.33, 305: q 0.33, 306: q 0.33, 307: q 0.33, 308: q 0.33, 309: q 0.33, 310: q 0.33

Regenerate with `tools/import_b1k_classical.py` (the commands are in the repository README).
