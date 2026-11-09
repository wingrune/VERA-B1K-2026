# tidying_living_room (2026 task 53)

Classical controller on RGB-D + proprioception (no learned parts) from the b1k_rl_guys repository, served by `solutions/b1k_classical/backend.py`.

- Source: `tasks/tidying_living_room/experiments/candidate/v02` at `84354487f4` (branch `pick-and-place`), source fingerprint `4d15dd994671692e…` (21 files), in `controller/`
- Inputs: head and wrist RGB-D, proprioception, robot-relative camera poses; static priors in `controller/priors/` (robot model, a map of the task's scene, object models); no instance pose and no simulator state
- Source documentation, scored runs and reports: b1k_rl_guys `tasks/tidying_living_room/`
- Measured (native evaluator, fresh process per episode, instances 301-310): SR 0/10, mean Q 0.23; per instance 301: q 0.50, 302: q 0.25, 303: q 0.00, 304: q 0.25, 305: q 0.25, 306: q 0.25, 307: q 0.25, 308: q 0.25, 309: q 0.00, 310: q 0.25

Regenerate with `tools/import_b1k_classical.py` (the commands are in the repository README).
