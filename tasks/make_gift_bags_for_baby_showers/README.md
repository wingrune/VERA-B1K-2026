# make_gift_bags_for_baby_showers (2026 task 86)

Classical controller on RGB-D + proprioception (no learned parts) from the b1k_rl_guys repository, served by `solutions/b1k_classical/backend.py`.

- Source: `tasks/make_gift_bags_for_baby_showers/experiments/candidate/m03` at `30eae01650` (branch `pick-and-place`), source fingerprint `679a05fdad4c9184…` (20 files), in `controller/`
- Inputs: head and wrist RGB-D, proprioception, robot-relative camera poses; static priors in `controller/priors/` (robot model, a map of the task's scene, object models); no instance pose and no simulator state
- Design and task facts: `controller/README.md`; scored runs and reports: b1k_rl_guys `tasks/make_gift_bags_for_baby_showers/`
- Measured (native evaluator, fresh process per episode, instances 301-310): SR 0/10, mean Q 0.18; per instance 301: q 0.17, 302: q 0.17, 303: q 0.17, 304: q 0.17, 305: q 0.17, 306: q 0.00, 307: q 0.33, 308: q 0.17, 309: q 0.17, 310: q 0.33

Regenerate with `tools/import_b1k_classical.py` (the commands are in the repository README).
