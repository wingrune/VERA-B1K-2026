# hiding_Easter_eggs candidate v40

v40 = frozen full_v38 + an exploration lure toward a strongly seen parked basket (MAKE_v39.py) + an egg view that never saw the basket moves the base and looks again (MAKE_v40.py).

- Entry point: `observation_policy.Policy` (batched observations, `--num-envs 1`; returns a [1, 23] R1Pro action). `easter_policy.EasterPolicy` is the
  controller; `Policy` only drops the batch dimension, as the scored runs' adapter did.
- Inputs: head and wrist RGB + `depth_linear`, the 61-D proprioception and the evaluator's robot-relative camera poses (`allowed()` in
  `easter_policy.py` drops every other key). Static priors: the robot model and object-local geometry / appearance constants in the code.
  No instance pose, no simulator state. `diagnostics.py`, `run_episode.py`, `run_native.py`, `probe_policy.py` are development harness files
  outside the controller's import closure.
- Scored (v3.9.3-post1, unmodified `python -m omnigibson.eval.eval`, RGBDFullResWrapper, bundled r1pro.yaml, `--write-video`, default timeout,
  one rollout, fresh policy server and evaluator per instance): SR 4/10, mean Q 0.478; 301: ok, 302: ok, 303: q 0.00, 304: q 0.00, 305: q 0.78, 306: ok, 307: q 0.00, 308: q 0.00, 309: ok, 310: q 0.00.
- Task facts, development history and diagnosis: `tasks/hiding_Easter_eggs/README.md`, `DEVELOPMENT_HISTORY.md`, `DIAGNOSIS.md`.
