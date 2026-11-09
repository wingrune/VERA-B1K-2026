# putting_dirty_dishes_in_sink (2026 task 85)

Classical controller on RGB-D + proprioception (no learned parts) from the b1k_rl_guys repository, served by `solutions/b1k_classical/backend.py`.

- Source: `tasks/putting_dirty_dishes_in_sink/experiments/candidate/k02` at `90b1ed1d17` (branch `pick-and-place`), source fingerprint `04f2286f4bcd35be…` (20 files), in `controller/`
- Inputs: head and wrist RGB-D, proprioception, robot-relative camera poses; static priors in `controller/priors/` (robot model, a map of the task's scene, object models); no instance pose and no simulator state
- Source documentation, scored runs and reports: b1k_rl_guys `tasks/putting_dirty_dishes_in_sink/`
- Measured (native evaluator, fresh process per episode, instances 301-310): SR 2/10, mean Q 0.57; per instance 301: ok, 302: ok, 303: q 0.25, 304: q 0.75, 305: q 0.75, 306: q 0.50, 307: q 0.75, 308: q 0.25, 309: q 0.50, 310: q 0.00

Regenerate with `tools/import_b1k_classical.py` (the commands are in the repository README).
