# collecting_aluminum_cans (2026 task 87)

Frozen classical RGB-D controller **a023**, served by `solutions/frozen_rgbd/backend.py` in the pinned CPU environment.

- Native instances 301–310, rollout 0: mean Q **0.850000**, success **3/10**.
- Original controller fingerprint: `e81d0a5903f980962e46e8db833d1da146fc6802a3e1346d35eea8effe207236`; 16 files in `controller/`, unchanged byte-for-byte.
- Evaluated image: `sha256:7f574effe873d8635c74c59baa6c0f19ccf80de866880f31895a7bc7533f6bea`. A rebuilt team image has a different identity.
- Input: RGB, depth, proprioception and robot-relative camera calibration. Task ID is checked by the adapter and excluded from policy inputs.
- Static object/robot/scene priors remain in the frozen package. No simulator imports are used by the policy runtime. Legacy serve_policy.py files, where present, are not entrypoints.
- One fresh backend process per task and episode; action float32 (1,23). Cans receives unbatched arrays; the other four preserve the evaluated input arrays.
- `fingerprint.json` retains the original compact-JSON digest convention. The existing git-archive importer uses a different convention; do not replace this fingerprint.
- Native results/videos are external HF artifacts, not source files. Full per-instance metrics and original source provenance are in solution.json and the prepared artifact manifest.

No manipulation, perception, motion planning, joint limits, collision checking or controller constants were changed during packaging.
The shared CPU dependency pins are in `solutions/frozen_rgbd/requirements.txt`; Torch 2.7.0 CPU is installed separately by Docker.
