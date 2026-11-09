# make_microwave_popcorn (2026 task 40)

Frozen native OmniGibson candidate **popcorn_p31**. The planner file is byte-identical to the controller used for the recorded evaluation.

- Native public instances 301–310, one rollout each: mean Q **0.600**, SR **6/10**.
- Source SHA-256: `7dea9dbcd87ce98109aad9d48826feae45ef7c139f3cae443fb32d32b4d6a183`.
- Base Git revision: `75d068ce11da9ff04646ea601cefbc48b326c7cc`; upstream H100-port provenance remains in `solution.json`.
- Original [JSON and MP4](https://huggingface.co/datasets/VERA-VLA/b1k_final_submission/tree/352087c61e891226d94263fa23e83601565a3333/standard.public.VERA_team.team.20261015).
- Scored policy image digest: `sha256:c78de5758f20661bf666c122809177ec4815f1d4b4f301d3ede8975002696de7`. This is the original per-task image, not a claim that a subsequently rebuilt combined image has the same digest.

## Runtime and routing

The common router selects `planner_cpu` and the existing `solutions/planner/backend.py`. This uses `$FROZEN_RGBD_PYTHON` (`/opt/venv-frozen-rgbd/bin/python`) already provided by the team Dockerfile, preserving the scored NumPy 1.26.0, SciPy 1.15.3 and Torch 2.7.0 CPU dependencies. No checkpoint or external service is required. Other tasks keep their existing routes and Python environments.

Inputs are onboard RGB-D, proprioception and camera calibration. The policy returns finite float32 23D R1Pro actions. No privileged simulator state is used for decisions.

Build and start the team image using the repository's existing instructions. Then, from BEHAVIOR-1K v3.9.3-post1:

```sh
python -m omnigibson.eval.eval --task-name make_microwave_popcorn \
  --host 127.0.0.1 --port 8000 --instance-indices 0 --num-envs 1 \
  --num-rollouts 1 --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
  --output-dir NEW_EMPTY_OUTPUT --write-video
```

Use public indices 0–9 for instances 301–310, one fresh simulator and independent policy state per episode. The recorded series used unchanged bundled R1Pro, three RGB-D cameras, default physics, horizon and timeouts. Original outputs passed the strict result checker and full MP4 decoding.
