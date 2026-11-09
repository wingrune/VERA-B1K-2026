# boxing_books_up_for_storage (2026 task 23)

Frozen classical RGB-D controller **books_b01**, served by `solutions/frozen_rgbd/backend.py` in the pinned CPU environment.

- Native instances 301–310, rollout 0: mean Q **0.183333**, success **0/10**.
- Original controller fingerprint: `e5c0cf37d385aa51c4a242f10bd06360c5dcb745a5b8e0417be6c39ae387b02d`; 77 files in `controller/`, unchanged byte-for-byte.
- Evaluated image: `sha256:7b220db539594d07473e61f2d88de1808f2805cc081454c52cb5251e0f254dd0`. A rebuilt team image has a different identity.
- Input: RGB, depth, proprioception and robot-relative camera calibration. Task ID is checked by the adapter and excluded from policy inputs.
- Static object/robot/scene priors remain in the frozen package. No simulator imports are used by the policy runtime. Legacy serve_policy.py files, where present, are not entrypoints.
- One fresh backend process per task and episode; action float32 (1,23). Cans receives unbatched arrays; the other four preserve the evaluated input arrays.
- `fingerprint.json` retains the original compact-JSON digest convention. The existing git-archive importer uses a different convention; do not replace this fingerprint.
- Native results/videos are external HF artifacts, not source files. Full per-instance metrics and original source provenance are in solution.json and the prepared artifact manifest.

No manipulation, perception, motion planning, joint limits, collision checking or controller constants were changed during packaging.
The shared CPU dependency pins are in `solutions/frozen_rgbd/requirements.txt`; Torch 2.7.0 CPU is installed separately by Docker.
