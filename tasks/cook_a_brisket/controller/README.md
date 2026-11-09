# cook_a_brisket candidate v20_submit

frozen full_v20 (tasks/cook_a_brisket/code, byte-identical).

- Entry point: `observation_policy.Policy` (batched observations, `--num-envs 1`; returns a [1, 23] R1Pro action). `brisket_policy.BrisketPolicy` is the
  controller; `Policy` only drops the batch dimension, as the scored runs' adapter did.
- Inputs: head and wrist RGB + `depth_linear`, the 61-D proprioception and the evaluator's robot-relative camera poses (`allowed()` in
  `brisket_policy.py` drops every other key). Static priors: the robot model and object-local geometry / appearance constants in the code.
  No instance pose, no simulator state. `diagnostics.py`, `run_episode.py`, `run_native.py`, `probe_policy.py` are development harness files
  outside the controller's import closure.
- Scored (v3.9.3-post1, unmodified `python -m omnigibson.eval.eval`, RGBDFullResWrapper, bundled r1pro.yaml, `--write-video`, default timeout,
  one rollout, fresh policy server and evaluator per instance): SR 2/10, mean Q 0.267; 301: q 0.00, 302: q 0.33, 303: ok, 304: q 0.00, 305: q 0.00, 306: ok, 307: q 0.00, 308: q 0.00, 309: q 0.00, 310: q 0.33.
- Task facts, development history and diagnosis: `tasks/cook_a_brisket/README.md`, `DEVELOPMENT_HISTORY.md`, `DIAGNOSIS.md`.
