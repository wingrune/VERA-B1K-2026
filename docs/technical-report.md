# VERA-Volve for the 2026 BEHAVIOR Challenge

System architecture and reproducible public validation

VERA team | November 9, 2026

## Abstract

VERA-Volve is a hybrid embodied policy that assigns household tasks to task-specific motion planners, classical RGB-D controllers and vision-language-action models. A fixed routing table covers all 100 challenge tasks. One WebSocket policy server manages the selected specialist, enforces the R1Pro action contract and isolates episode failures. The release preserves the original solution code and publishes instructions for recovering all five routed checkpoints from the public, digest-pinned submission image. We summarize an immutable public artifact snapshot containing 820 native-evaluator episodes across 82 tasks. Across those published tasks, mean final Q is 0.531441 and 259 episodes report success. These measurements describe a ten-instance public slice, not a verified official leaderboard score. The release supports inspection and inference reproduction; complete COMET training and evolution-search reproduction require materials beyond this source snapshot.

## 1. Task and system setting

The target is the 2026 BEHAVIOR Challenge with the R1Pro robot under BEHAVIOR-1K v3.9.3-post1. The policy receives observations through the evaluator protocol and returns a 23-dimensional robot action. Geometric policies use onboard RGB-D, proprioception and robot-relative camera information, together with static robot, object or scene priors. VLA routes use camera RGB and proprioception; the COMET adapter also conditions on the task name.

The routing table is fixed before execution. It does not select a backend from privileged live simulator state. The following counts describe the released configuration, excluding the wildcard fallback row.

| Backend | Tasks | Function |
|---|---:|---|
| planner | 8 | Task-specific planning programs |
| planner_cpu | 2 | Native CPU-environment task planners |
| b1k_classical | 25 | Geometric classical controllers |
| frozen_rgbd | 5 | Preserved controllers with pinned environment |
| larchenko_ckpt2 / ckpt3 | 11 / 3 | Adapted 2025 VLA policies |
| comet2 / comet2_1 / comet3 | 7 / 6 / 33 | Team pi0.5 checkpoints |

<!-- pagebreak -->

## 2. Policy architecture

### 2.1 Task routing and backend lifecycle

The evaluator connects to port 8000. The server reads the observation's task ID, resolves the corresponding route in `config/routing.csv`, and starts the backend command registered in `config/solutions.json`. At most one backend is active. Switching specialists stops the previous process, which bounds the policy-side memory footprint. A second evaluator connection waits until the first closes; independent simultaneous evaluations therefore require separate policy servers.

Reset behavior is explicit in the backend registry. Planner, classical, frozen and COMET routes restart their backend per episode. Larchenko routes reuse their checkpoint process and reset policy state through the adapter. The router verifies finite 23D actions and, when supplied, finite action chunks with consistent first actions. A backend error, invalid reply or timeout fails the current episode: the server returns an action that holds the current robot pose until the next reset. This permits later episodes to proceed without silently changing the selected method.

### 2.2 Task planners and geometric controllers

Task folders contain the selected planning programs, entrypoint metadata and provenance. Some planner records describe programs selected by evolution; their recorded H100/Isaac Lab port measurements are distinct from native simulator validation. Other folders package classical controllers with geometric perception, whole-body IK, sampling-based motion planning and static priors. cuRobo supplies GPU IK for planners that depend on it.

The release serves popcorn and smoke-detector planners in a pinned CPU environment. A separate set of five frozen RGB-D controllers retains its original source fingerprints and dependency pins. These choices preserve the numerical and import environments expected by the packaged policies. The generic backend adapters translate evaluator observations and episode resets to each controller's API while retaining the controller code.

### 2.3 VLA specialists

The adapted Larchenko backend vendors the 2025 BEHAVIOR solution and maps task identities into its 2025 policy interface. Its two routed checkpoints serve 14 tasks. The source documents adapter differences, including action conversion and a known base-velocity-frame gap; the release does not assert perfect behavioral equivalence to the original 2025 system.

The team COMET checkpoints are group2_hour_a01 step 2498, group1_7h_a01 step 20408 and all63_1h_a01 step 2540. They share the bundled OpenPI runtime but serve different task sets. COMET takes three RGB camera views, 61D R1Pro proprioception and the task-name prompt. The model predicts a 32-step sequence; the existing scheduler preserves its 26-to-20 compression, gripper handling and replanning after 20 executed actions. The evaluator must request 20-action replay. The fallback wildcard route uses comet3 for task IDs not otherwise listed, with an empty prompt for unknown IDs.

<!-- pagebreak -->

## 3. Public validation evidence

The evidence snapshot is VERA-VLA/b1k_final_submission revision `a7e21a893581e5f6a1bf71d8a0e9780530ecf241`, submission folder `standard.public.VERA_team.team.20261015`. It contains 820 JSON episodes for 82 tasks, with ten distinct instances per task: 301-310, rollout 0. Every JSON has a corresponding video path in the dataset tree. The remaining 18 task IDs have no episodes in this snapshot.

The summary is computed from committed JSON blobs. For task t with ten published episodes, mean Q is the arithmetic mean of `q_score.final`. The published-task aggregate is the equally weighted mean of the 82 task means. Successes count the evaluator's explicit Boolean `success`, rather than inferring success from a rounded Q. This gives mean Q 0.531441 and 259 successful episodes out of 820. These values are descriptive statistics for the available slice and are not reported as the official challenge score.

| Task | Successful episodes | Mean final Q |
|---|---:|---:|
| Installing smoke detectors | 10 / 10 | 1.000000 |
| Installing a fax machine | 10 / 10 | 1.000000 |
| Loading the car | 10 / 10 | 1.000000 |
| Picking up trash | 9 / 10 | 0.966667 |
| Setting mousetraps | 0 / 10 | 0.966667 |
| Collecting aluminum cans | 3 / 10 | 0.850000 |
| Turning on radio | 8 / 10 | 0.800000 |
| Attaching a camera to a tripod | 6 / 10 | 0.600000 |
| Making rose centerpieces | 0 / 10 | 0.400000 |

### 3.1 Interpretation and provenance

The examples illustrate the difference between partial completion and full task success. Setting mousetraps has high mean Q but no successful episodes in the published set. A video of a successful instance is not a task-level success estimate; the report's counts use all ten published instances.

The released routing table is not a complete per-episode policy provenance record. Some artifacts were collected before tasks moved to newer routes. Per-task README files and solution metadata preserve earlier candidate identities, source commits, fingerprints and hardware-specific measurements where available. The summary labels the current configuration as a release route and does not attribute every historical episode to that route.

The evidence tool checks task identities, filenames, Q ranges, success consistency, the complete 301-310 instance sets and video-path presence. Its manifest records SHA256 hashes of every JSON. All video payloads were not downloaded or decoded during release preparation; file-tree pairing is the scope of the 820-video presence check. One radio video was downloaded, matched to its LFS SHA256 and used to derive the blog poster.

<!-- pagebreak -->

## 4. Reproduction and release artifacts

### 4.1 Recover the packaged inference environment

The publicly readable Linux/amd64 image is pinned by the following OCI digest. It contains policy source, runtime dependencies, routing and all five model checkpoint roots. Registry metadata shows 64 layers and approximately 71.1 GiB compressed content. The image is an available weight distribution even though the public VERA-VLA listing does not expose separate COMET model repositories.

```text
ghcr.io/alstar8/b1k-evo-submit@sha256:
1ab7d94f5b3ecc347dbb9adccc2d9212cfbd2ed4aad078bda82faad607a240c5
```

Run the policy on Linux with Docker, NVIDIA Container Toolkit and sufficient GPU resources. The submission specifies 24 GB VRAM for one active policy backend; the native simulator needs additional resources. Install BEHAVIOR-1K v3.9.3-post1 and its licensed assets separately. The policy requires no external inference service or policy network connection at runtime.

Use one native evaluator process per instance, `--num-envs 1`, one rollout and default timeouts. Select DefaultWrapper for Larchenko routes. Select RGBDFullResWrapper for the other released routes. COMET routes require replay chunk size 20; other routes use 1. The reproduction guide includes commands that derive these choices from the released image's own routing and registry files. Preserve JSON, MP4, image identity and logs for each new run.

### 4.2 Rebuild and inspect the weights

The extraction helper pulls the pinned image, creates a stopped container and copies checkpoints into the source layout. It does not start the policy or require a GPU. Copies are staged, existing destinations are protected, and the group2 COMET files are verified against the included checksum manifest. Its aggregate fingerprint is recorded below.

```text
9afdb712de022c058fc10495f8aca921c24b38291dd40aca9950c38805799923
```

The group1 and all63 checkpoints lack equivalent independent per-file manifests in the source release; their available distribution is pinned by the image digest. The image builder copies each routed checkpoint once through named build contexts and keeps backend dependencies in separate environments. Some installer or base-image references still depend on external registries, so a fresh source build is not claimed to be bit-for-bit identical to the prebuilt image.

### 4.3 Verification performed for this release

The source-preservation check verifies 1,832 imported file entries, including symlink targets and executable modes, against the original commit. The inherited CPU suite passes 28 tests, with four optional frozen-reference differential tests skipped because external fixtures are absent. The bundled OpenPI client must be on PYTHONPATH for transport tests. Routing and frozen-controller fingerprint checks can run without weights. The static blog was checked at desktop and mobile viewport sizes, and task search and missing-evidence filtering were exercised.

Release preparation verified public image access and checkpoint layer metadata. It did not download the full 71.1 GiB image, execute full checkpoint extraction, rebuild the GPU image or perform fresh native simulator rollouts. Recorded validation results remain the evidence described in Section 3.

<!-- pagebreak -->

## 5. Limitations and lessons

The released package is an inference and evidence release. Vendored runtime and training-related modules do not constitute a complete COMET training recipe: the release does not provide a fully specified data selection, initial checkpoint, training command and compute record for each of the three team models. Similarly, selected evolved programs and provenance records do not reproduce the entire program-search history. These are meaningful limits for reproducing how the specialists were obtained.

Coverage must be separated from measured performance. All 100 tasks are routed, but the pinned public snapshot contains 82 task sets. Some task notes report strong native outcomes while other tasks show partial completion or failure. Measurements from an Isaac Lab port and the official native evaluator are different experiments; the release preserves their identities and does not pool them.

The hybrid design makes interface details consequential. Wrapper selection controls observations. Action scheduling changes execution semantics. Process isolation and resets determine whether one episode affects the next. Numerical dependencies and static priors are part of the behavior of geometric controllers. Recovering the full policy package, rather than only a model file, is therefore the practical starting point for reproduction.

Model, tokenizer, simulator and dataset terms remain separate from the root Apache-2.0 project license. The root license and all bundled notices are preserved. Simulator assets are obtained through the upstream installation and are not redistributed in this source release.

## References and artifact index

1. [BEHAVIOR Challenge 2026](https://behavior.stanford.edu/challenge/index.html), including its evaluation and submission documentation.

2. [BEHAVIOR-1K v3.9.3-post1](https://github.com/StanfordVL/BEHAVIOR-1K/tree/v3.9.3-post1), the evaluator version targeted by this package.

3. [VERA-Volve source release](https://github.com/wingrune/VERA-B1K-2026). Imported source commit: 309bb650dc1606319a4309d5b0ab64702f8ad6ab from alstar8/b1k_evo_submit. The release manifest records preserved entries.

4. [Pinned VERA validation snapshot](https://huggingface.co/datasets/VERA-VLA/b1k_final_submission/tree/a7e21a893581e5f6a1bf71d8a0e9780530ecf241). JSON hashes, per-task statistics and video paths are published with the code.

5. [Larchenko et al., BEHAVIOR solution](https://github.com/IliaLarchenko/behavior-1k-solution) and [technical report](https://arxiv.org/abs/2512.06951). The adapted implementation and upstream license are vendored in this release.

6. [OpenPI](https://github.com/Physical-Intelligence/openpi), [cuRobo](https://github.com/NVlabs/curobo), and [b1k_rl_guys](https://github.com/tttonyalpha/b1k_rl_guys). Task-level provenance records and bundled licenses identify their use in the solution.

The accompanying repository contains the full reproduction guide, checkpoint extraction helper, editable report source and GitHub Pages article. The report's claims are scoped to the released implementation and pinned public artifacts; no new training or simulator experiments are asserted.
