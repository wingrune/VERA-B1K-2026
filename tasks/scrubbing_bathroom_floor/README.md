# scrubbing_bathroom_floor (2026 task 60)

Classical controller on RGB-D + proprioception (no learned parts) from the b1k_rl_guys repository, served by `solutions/b1k_classical/backend.py`.

- Source: `tasks/scrubbing_bathroom_floor/experiments/candidate/b01` at `9d5cb69e2e` (branch `fax-manip`), source fingerprint `e50724530b286555…` (30 files), in `controller/`
- Inputs: head and wrist RGB-D, proprioception, robot-relative camera poses; static priors in `controller/priors/` (robot model, a map of the task's scene, object models); no instance pose and no simulator state
- Design and task facts: `controller/README.md`; scored runs and reports: b1k_rl_guys `tasks/scrubbing_bathroom_floor/`
- Measured (native evaluator, fresh process per episode, instances 301-310): SR 2/10, mean Q 0.20; per instance 301: q 0.00, 302: ok, 303: ok, 304: q 0.00, 305: q 0.00, 306: q 0.00, 307: q 0.00, 308: q 0.00, 309: q 0.00, 310: q 0.00

Regenerate with `tools/import_b1k_classical.py` (the commands are in the repository README).
