# store_batteries (2026 task 82)

Classical controller on RGB-D + proprioception (no learned parts) from the b1k_rl_guys repository, served by `solutions/b1k_classical/backend.py`.

- Source: `tasks/store_batteries/experiments/candidate/x02` at `63d6172b9e` (branch `pick-and-place`), source fingerprint `da6fd434df7d0cc3…` (21 files), in `controller/`
- Inputs: head and wrist RGB-D, proprioception, robot-relative camera poses; static priors in `controller/priors/` (robot model, a map of the task's scene, object models); no instance pose and no simulator state
- Design and task facts: `controller/README.md`; scored runs and reports: b1k_rl_guys `tasks/store_batteries/`
- Measured (native evaluator, fresh process per episode, instances 301-310): SR 6/10, mean Q 0.80; per instance 301: q 0.67, 302: q 0.33, 303: ok, 304: q 0.67, 305: ok, 306: q 0.33, 307: ok, 308: ok, 309: ok, 310: ok

Regenerate with `tools/import_b1k_classical.py` (the commands are in the repository README).
