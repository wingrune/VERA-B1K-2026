# putting_away_Halloween_decorations (2026 task 2)

Classical controller on RGB-D + proprioception (no learned parts) from the b1k_rl_guys repository, served by `solutions/b1k_classical/backend.py`.

- Source: `tasks/putting_away_Halloween_decorations/experiments/candidate/h01` at `9d5cb69e2e` (branch `fax-manip`), source fingerprint `bc729e603e68c6aa…` (24 files), in `controller/`
- Inputs: head and wrist RGB-D, proprioception, robot-relative camera poses; static priors in `controller/priors/` (robot model, a map of the task's scene, object models); no instance pose and no simulator state
- Design and task facts: `controller/README.md`; scored runs and reports: b1k_rl_guys `tasks/putting_away_Halloween_decorations/`
- Measured (native evaluator, fresh process per episode, instances 301-310): SR 9/10, mean Q 0.97; per instance 301: ok, 302: ok, 303: ok, 304: ok, 305: ok, 306: q 0.71, 307: ok, 308: ok, 309: ok, 310: ok

Regenerate with `tools/import_b1k_classical.py` (the commands are in the repository README).
