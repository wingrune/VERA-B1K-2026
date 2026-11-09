# sorting_bottles_cans_and_paper (2026 task 52)

Classical controller on RGB-D + proprioception (no learned parts) from the b1k_rl_guys repository, served by `solutions/b1k_classical/backend.py`.

- Source: `tasks/sorting_bottles_cans_and_paper/experiments/candidate/o01` at `ede53534c6` (branch `pick-and-place`), source fingerprint `77468038f6be24d9…` (20 files), in `controller/`
- Inputs: head and wrist RGB-D, proprioception, robot-relative camera poses; static priors in `controller/priors/` (robot model, a map of the task's scene, object models); no instance pose and no simulator state
- Design and task facts: `controller/README.md`; scored runs and reports: b1k_rl_guys `tasks/sorting_bottles_cans_and_paper/`
- Measured (native evaluator, fresh process per episode, instances 301-310): SR 0/10, mean Q 0.09; per instance 301: q 0.06, 302: q 0.06, 303: q 0.00, 304: q 0.12, 305: q 0.06, 306: q 0.06, 307: q 0.12, 308: q 0.12, 309: q 0.12, 310: q 0.12

Regenerate with `tools/import_b1k_classical.py` (the commands are in the repository README).
