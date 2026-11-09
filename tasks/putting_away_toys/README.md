# putting_away_toys (2026 task 54)

Classical controller on RGB-D + proprioception (no learned parts) from the b1k_rl_guys repository, served by `solutions/b1k_classical/backend.py`.

- Source: `tasks/putting_away_toys/experiments/candidate/y01` at `3be4f6ae47` (branch `pick-and-place`), source fingerprint `fbe0742fd913732b…` (20 files), in `controller/`
- Inputs: head and wrist RGB-D, proprioception, robot-relative camera poses; static priors in `controller/priors/` (robot model, a map of the task's scene, object models); no instance pose and no simulator state
- Design and task facts: `controller/README.md`; scored runs and reports: b1k_rl_guys `tasks/putting_away_toys/`
- Measured (native evaluator, fresh process per episode, instances 301-310): SR 4/10, mean Q 0.79; per instance 301: q 0.75, 302: q 0.88, 303: q 0.00, 304: q 0.88, 305: q 0.50, 306: ok, 307: ok, 308: ok, 309: ok, 310: q 0.88

Regenerate with `tools/import_b1k_classical.py` (the commands are in the repository README).
