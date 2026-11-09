# make_rose_centerpieces (2026 task 56)

Classical controller on RGB-D + proprioception (no learned parts) from the b1k_rl_guys repository, served by `solutions/b1k_classical/backend.py`.

- Source: `tasks/make_rose_centerpieces/m15_validation_20261016/controller` at `2b611cb862` (branch `pick-and-place-wingrune`), source fingerprint `4b7ffca7ea4b1046…` (235 files), in `controller/`
- Inputs: head and wrist RGB-D, proprioception, robot-relative camera poses; static priors in `controller/controller/priors/` (robot model, a map of the task's scene, object models); no instance pose and no simulator state
- Source documentation, scored runs and reports: b1k_rl_guys `tasks/make_rose_centerpieces/`
- Adapter: `solutions/b1k_classical/rose_m15.py`, frozen M15/h32 with CUDA perception. Runtime code and static priors retain their source bytes; historical variants and evaluation artifacts are excluded. The controller manifest covers the retained files. Evaluation reports and videos are stored outside this submission repository.
- Measured (native evaluator, fresh process per episode, instances 301-310): SR 0/10, mean Q 0.40; per instance 301: q 0.75, 302: q 0.50, 303: q 0.75, 304: q 0.25, 305: q 0.00, 306: q 0.50, 307: q 0.50, 308: q 0.50, 309: q 0.00, 310: q 0.25

Regenerate with `tools/import_b1k_classical.py` (the commands are in the repository README).
