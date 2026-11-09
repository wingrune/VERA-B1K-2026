# clearing_food_from_table_into_fridge (2026 task 25)

Classical controller on RGB-D + proprioception (no learned parts) from the b1k_rl_guys repository, served by `solutions/b1k_classical/backend.py`.

- Source: `tasks/clearing_food_from_table_into_fridge/experiments/candidate/z02` at `fdab482633` (branch `pick-and-place`), source fingerprint `65fae6c0b95ae889…` (27 files), in `controller/`
- Inputs: head and wrist RGB-D, proprioception, robot-relative camera poses; static priors in `controller/priors/` (robot model, a map of the task's scene, object models); no instance pose and no simulator state
- Source documentation, scored runs and reports: b1k_rl_guys `tasks/clearing_food_from_table_into_fridge/`
- Measured (native evaluator, fresh process per episode, instances 301-310): SR 0/10, mean Q 0.36; per instance 301: q 0.20, 302: q 0.40, 303: q 0.40, 304: q 0.40, 305: q 0.40, 306: q 0.40, 307: q 0.20, 308: q 0.40, 309: q 0.40, 310: q 0.40

Regenerate with `tools/import_b1k_classical.py` (the commands are in the repository README).
