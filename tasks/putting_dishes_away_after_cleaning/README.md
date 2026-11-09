# putting_dishes_away_after_cleaning (2026 task 11)

Classical controller on RGB-D + proprioception (no learned parts) from the b1k_rl_guys repository, served by `solutions/b1k_classical/backend.py`.

- Source: `tasks/putting_dishes_away_after_cleaning/experiments/candidate/a01` at `7906a45943` (branch `pick-and-place`), source fingerprint `4dbeb84693eefc00…` (25 files), in `controller/`
- Inputs: head and wrist RGB-D, proprioception, robot-relative camera poses; static priors in `controller/priors/` (robot model, a map of the task's scene, object models); no instance pose and no simulator state
- Design and task facts: `controller/README.md`; scored runs and reports: b1k_rl_guys `tasks/putting_dishes_away_after_cleaning/`
- Measured (native evaluator, fresh process per episode, instances 301-310): SR 0/10, mean Q 0.19; per instance 301: q 0.29, 302: q 0.21, 303: q 0.29, 304: q 0.14, 305: q 0.14, 306: q 0.21, 307: q 0.21, 308: q 0.14, 309: q 0.29, 310: q 0.00

Regenerate with `tools/import_b1k_classical.py` (the commands are in the repository README).
