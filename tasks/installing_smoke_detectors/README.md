# installing_smoke_detectors (2026 task 63)

Frozen native OmniGibson candidate **smoke_s05**. The planner file is byte-identical to the controller used for the recorded evaluation.

- Native public instances 301–310, one rollout each: mean Q **1.000**, SR **10/10**.
- Source SHA-256: `26e5db3513f459b182d1731e1e5d7ba75940caab46a1f2b7f1f81cf13d5d0554`.
- Base Git revision: `4cb848e6c05bd5231676699b72b8a7ca71557673`; upstream H100-port provenance remains in `solution.json`.
- Original [JSON and MP4](https://huggingface.co/datasets/VERA-VLA/b1k_final_submission/tree/352087c61e891226d94263fa23e83601565a3333/standard.public.VERA_team.team.20261015).
- Scored policy image digest: `sha256:d5e525b0ab25d3da55a029154ffb4deb12d6ebe29a7b285d78a6625ae141c10d`. This is the original per-task image, not a claim that a subsequently rebuilt combined image has the same digest.

## Runtime and routing

The common router selects `planner_cpu` and the existing `solutions/planner/backend.py`. This uses `$FROZEN_RGBD_PYTHON` (`/opt/venv-frozen-rgbd/bin/python`) already provided by the team Dockerfile, preserving the scored NumPy 1.26.0, SciPy 1.15.3 and Torch 2.7.0 CPU dependencies. No checkpoint or external service is required. Other tasks keep their existing routes and Python environments.

Inputs are onboard RGB-D, proprioception and camera calibration. The policy returns finite float32 23D R1Pro actions. No privileged simulator state is used for decisions.

Build and start the team image using the repository's existing instructions. Then, from BEHAVIOR-1K v3.9.3-post1:

```sh
python -m omnigibson.eval.eval --task-name installing_smoke_detectors \
  --host 127.0.0.1 --port 8000 --instance-indices 0 --num-envs 1 \
  --num-rollouts 1 --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
  --output-dir NEW_EMPTY_OUTPUT --write-video
```

Use public indices 0–9 for instances 301–310, one fresh simulator and independent policy state per episode. The recorded series used unchanged bundled R1Pro, three RGB-D cameras, default physics, horizon and timeouts. Original outputs passed the strict result checker and full MP4 decoding.
