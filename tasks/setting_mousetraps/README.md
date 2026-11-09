# setting_mousetraps (2026 task 5)

Classical controller on RGB-D + proprioception (no learned parts) from the b1k_rl_guys repository, served by `solutions/b1k_classical/backend.py`.

- Source: `tasks/setting_mousetraps/experiments/candidate/v01` at `4b0dafd310` (branch `pick-and-place`), source fingerprint `132794e60df66452…` (21 files), in `controller/`
- Inputs: head and wrist RGB-D, proprioception, robot-relative camera poses; static priors in `controller/priors/` (robot model, a map of the task's scene, object models); no instance pose and no simulator state
- Source documentation, scored runs and reports: b1k_rl_guys `tasks/setting_mousetraps/`
- Measured (native evaluator, fresh process per episode, instances 301-310): SR 0/10, mean Q 0.97; per instance 301: q 1.00, 302: q 1.00, 303: q 1.00, 304: q 0.83, 305: q 0.83, 306: q 1.00, 307: q 1.00, 308: q 1.00, 309: q 1.00, 310: q 1.00

Regenerate with `tools/import_b1k_classical.py` (the commands are in the repository README).
