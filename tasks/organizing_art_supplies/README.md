# organizing_art_supplies (2026 task 59)

Frozen classical RGB-D controller **source87**, served by `solutions/frozen_rgbd/backend.py` in the pinned CPU environment.

- Native instances 301–310, rollout 0: mean Q **0.660000**, success **3/10**.
- Original controller fingerprint: `e32013f4ae8165c22f7ba7107be5e53be39ed5665add198ca24c7b9255b481ed`; 45 files in `controller/`, unchanged byte-for-byte.
- Evaluated image: `sha256:da14ad6165f4e809cda7c306a929be0d203ac4a24f12ab5d83d2e3e6142cc767`. A rebuilt team image has a different identity.
- Input: RGB, depth, proprioception and robot-relative camera calibration. Task ID is checked by the adapter and excluded from policy inputs.
- Static object/robot/scene priors remain in the frozen package. No simulator imports are used by the policy runtime. Legacy serve_policy.py files, where present, are not entrypoints.
- One fresh backend process per task and episode; action float32 (1,23). Cans receives unbatched arrays; the other four preserve the evaluated input arrays.
- `fingerprint.json` retains the original compact-JSON digest convention. The existing git-archive importer uses a different convention; do not replace this fingerprint.
- Native results/videos are external HF artifacts, not source files. Full per-instance metrics and original source provenance are in solution.json and the prepared artifact manifest.

No manipulation, perception, motion planning, joint limits, collision checking or controller constants were changed during packaging.
The shared CPU dependency pins are in `solutions/frozen_rgbd/requirements.txt`; Torch 2.7.0 CPU is installed separately by Docker.
Earlier direct-policy development Q 0.86 is not this native evaluation; the submitted native mean Q is 0.66.
