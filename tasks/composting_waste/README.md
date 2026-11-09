# composting_waste (2026 task 90)

Frozen classical RGB-D controller **c05**, served by `solutions/frozen_rgbd/backend.py` in the pinned CPU environment.

- Native instances 301–310, rollout 0: mean Q **0.650000**, success **3/10**.
- Original controller fingerprint: `db09993762620e483dfb353fb3bc005338adcfaa4f5bf2290e515e5026f4779b`; 50 files in `controller/`, unchanged byte-for-byte.
- Evaluated image: `sha256:71e755c971328cdc4acff6c8d7358c9459d41c40d0712bc6683a0d6456efe92f`. A rebuilt team image has a different identity.
- Input: RGB, depth, proprioception and robot-relative camera calibration. Task ID is checked by the adapter and excluded from policy inputs.
- Static object/robot/scene priors remain in the frozen package. No simulator imports are used by the policy runtime. Legacy serve_policy.py files, where present, are not entrypoints.
- One fresh backend process per task and episode; action float32 (1,23). Cans receives unbatched arrays; the other four preserve the evaluated input arrays.
- `fingerprint.json` retains the original compact-JSON digest convention. The existing git-archive importer uses a different convention; do not replace this fingerprint.
- Native results/videos are external HF artifacts, not source files. Full per-instance metrics and original source provenance are in solution.json and the prepared artifact manifest.

No manipulation, perception, motion planning, joint limits, collision checking or controller constants were changed during packaging.
The shared CPU dependency pins are in `solutions/frozen_rgbd/requirements.txt`; Torch 2.7.0 CPU is installed separately by Docker.
