# comet2

π0.5 backend for tasks 66, 71, 79, 84, 88, 97 and 98 (solution `comet2`, `checkpoints/comet2`). The same backend
with the group1 checkpoint is solution `comet2_1` (`checkpoints/comet2_1`, tasks 20, 26, 67, 81, 86 and 91), and with
the all63 checkpoint solution `comet3` (`checkpoints/comet3`): the tasks it was trained on that no other route serves
and, through the `*` routing row, any task id outside the 100 (empty prompt). All 100 tasks are in
`tasks.json`. Each checkpoint is supplied separately and must contain
`params/` and `assets/behavior-1k/2026-challenge-rgb224/norm_stats.json`.

```bash
ln -s /absolute/path/to/checkpoint checkpoints/comet2
python3 scripts/check_comet2_checkpoint.py checkpoints/comet2
python3 scripts/build_docker.py --tag b1k-evo-submit:comet2
docker run --rm --gpus 'device=0' -p 8000:8000 b1k-evo-submit:comet2
```

The checkpoint path can instead be set in the comet2 rows of `config/routing.csv`.
Other routed checkpoints must be available for a complete team build.

The builder creates `/opt/venv-comet2` using Python 3.11.13 and
`requirements.lock`. Runtime source and tokenizer are bundled in `runtime/`;
checksums are in `runtime_manifest.json`. No external policy image or inference
service is required. Other backends retain their separate environments.

Inputs are three RGB cameras, R1Pro 61D proprioception and the task name.
The model predicts 32 actions. The scheduler preserves the 26-to-20 compression,
gripper handling and replanning every 20 executed actions. The evaluator launcher
selects `--replay-action-chunk-size 20` for this backend. Use `--num-envs 1` and a
separate router per evaluator. Each episode reset starts a fresh backend process.

```bash
python3 server/routing.py --no-files
python3 scripts/build_docker.py --dry-run --no-files --tag b1k-evo-submit:comet2
/opt/venv-comet2/bin/python solutions/comet2/tools/check_runtime.py
/opt/venv-comet2/bin/python -m unittest discover -s tests -v
```

Differential tests additionally accept `COMET2_REFERENCE_BACKEND` and
`COMET2_OBSERVATION`. They compare saved observations and identical model output
chunks without requiring GPU inference.

Licenses and required third-party notices are retained in `licenses/`.
