# organizing_school_stuff (2026 task 73)

Frozen classical RGB-D controller **s91_integrated_contact_recovery**, served by `solutions/frozen_rgbd/backend.py` in the pinned CPU environment.

- Native instances 301–310, rollout 0: mean Q **0.800000**, success **3/10**.
- Original controller fingerprint: `37d9fd94fbbfa1054a78eda87df0a4931c8b9fa1f2a531cea29e880c1b6533d9`; 67 files in `controller/`, unchanged byte-for-byte.
- Archived reproduction image (packaged after the native series; verified by replay): `sha256:c7bc2129c3a4a929ab05014557644de235a01e362ff54dee1e54c54181786fb3`. A rebuilt team image has a different identity.
- Input: RGB, depth, proprioception and robot-relative camera calibration. Task ID is checked by the adapter and excluded from policy inputs.
- Static object/robot/scene priors remain in the frozen package. No simulator imports are used by the policy runtime. Legacy serve_policy.py files, where present, are not entrypoints.
- One fresh backend process per task and episode; action float32 (1,23). Cans receives unbatched arrays; the other four preserve the evaluated input arrays.
- `fingerprint.json` retains the original compact-JSON digest convention. The existing git-archive importer uses a different convention; do not replace this fingerprint.
- Native results/videos are external HF artifacts, not source files. Full per-instance metrics and original source provenance are in solution.json and the prepared artifact manifest.

No manipulation, perception, motion planning, joint limits, collision checking or controller constants were changed during packaging.
The shared CPU dependency pins are in `solutions/frozen_rgbd/requirements.txt`; Torch 2.7.0 CPU is installed separately by Docker.
The source series is named dev_s91_301_310, but these are original native evaluator JSON/video. Its evaluator-side diagnostic bootstrap is documented in artifact provenance; it did not supply simulator state to the policy.
