# dispose_of_glass (2026 task 76)

Classical controller on RGB-D + proprioception (no learned parts) from the b1k_rl_guys repository, served by `solutions/b1k_classical/backend.py`.

- Source: `tasks/dispose_of_glass/experiments/candidate/w01` at `1fb32b3583` (branch `pick-and-place`), source fingerprint `4c80a0f2b3d68cbc…` (20 files), in `controller/`
- Inputs: head and wrist RGB-D, proprioception, robot-relative camera poses; static priors in `controller/priors/` (robot model, a map of the task's scene, object models); no instance pose and no simulator state
- Design and task facts: `controller/README.md`; scored runs and reports: b1k_rl_guys `tasks/dispose_of_glass/`
- Measured (native evaluator, fresh process per episode, instances 301-310): SR 2/10, mean Q 0.65; per instance 301: q 0.50, 302: ok, 303: ok, 304: q 0.75, 305: q 0.50, 306: q 0.75, 307: q 0.25, 308: q 0.50, 309: q 0.75, 310: q 0.50

Regenerate with `tools/import_b1k_classical.py` (the commands are in the repository README).
