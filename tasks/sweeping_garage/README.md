# sweeping_garage (2026 task 57)

Classical controller on RGB-D + proprioception (no learned parts) from the b1k_rl_guys repository, served by `solutions/b1k_classical/backend.py`.

- Source: `tasks/sweeping_garage/experiments/candidate/s03` at `9d5cb69e2e` (branch `fax-manip`), source fingerprint `1d2ec7238bcf7d36…` (27 files), in `controller/`
- Inputs: head and wrist RGB-D, proprioception, robot-relative camera poses; static priors in `controller/priors/` (robot model, a map of the task's scene, object models); no instance pose and no simulator state
- Design and task facts: `controller/README.md`; scored runs and reports: b1k_rl_guys `tasks/sweeping_garage/`
- Measured (native evaluator, fresh process per episode, instances 301-310): SR 6/10, mean Q 0.70; per instance 301: q 0.00, 302: q 0.50, 303: ok, 304: ok, 305: q 0.50, 306: ok, 307: q 0.00, 308: ok, 309: ok, 310: ok

Regenerate with `tools/import_b1k_classical.py` (the commands are in the repository README).
