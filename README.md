# VERA-Volve · BEHAVIOR Challenge 2026

An open-source release of **VERA's hybrid solution** to the [2026 BEHAVIOR Challenge](https://behavior.stanford.edu/challenge/index.html): task-specific RGB-D planners and classical controllers, combined with π0.5 vision-language-action policies behind one Dockerized policy server.

**[Project blog](https://wingrune.github.io/VERA-B1K-2026/)** · **[Technical report](docs/technical-report.pdf)** · **[Reproduction guide](docs/reproduce.md)** · **[Validation JSONs and videos](https://huggingface.co/datasets/VERA-VLA/b1k_final_submission)** · **[Challenge leaderboard](https://huggingface.co/spaces/behavior-1k/2026-challenge-leaderboard)**

## What is released

| Material | Location |
|---|---|
| Policy server, routing, planners, controllers, VLA adapters and dependencies | This repository |
| All five routed model checkpoints and the packaged inference environments | [Public submission image](https://github.com/users/alstar8/packages/container/package/b1k-evo-submit); extract with `scripts/extract_release_checkpoints.py` |
| Team checkpoint identities and required directory layout | [Model weights guide](docs/weights.md) |
| Technical report with architecture, evidence and limitations | [PDF](docs/technical-report.pdf) · [editable source](docs/technical-report.md) |
| Public validation artifacts | [VERA-VLA/b1k_final_submission](https://huggingface.co/datasets/VERA-VLA/b1k_final_submission/tree/a7e21a893581e5f6a1bf71d8a0e9780530ecf241/standard.public.VERA_team.team.20261015) |
| Recomputed results and per-file evidence hashes | [Results data](docs/data/public-results.json) · [evidence manifest](docs/data/evidence-manifest.json) |
| Source provenance | [Release manifest](release/provenance.json) · [preserved-file hashes](release/source-manifest.json) |

The code was imported from [`alstar8/b1k_evo_submit`](https://github.com/alstar8/b1k_evo_submit) at commit `309bb650dc1606319a4309d5b0ab64702f8ad6ab`. All 1,832 imported entries other than the rewritten root README are preserved, including licenses, symlinks and executable permissions. Documentation and release helpers are added alongside them.

## The approach

Every observation carries a task ID. A fixed [routing table](config/routing.csv) selects one specialist; the [router](server/router.py) runs one backend at a time and returns a 23-dimensional R1Pro action through the evaluator's WebSocket protocol on port **8000**.

| Backend | Routed tasks | Role |
|---|---:|---|
| `planner` | 8 | Task-specific RGB-D planners, including evolved programs |
| `planner_cpu` | 2 | Native-validated popcorn and smoke-detector planners in the pinned CPU environment |
| `b1k_classical` | 25 | Geometric perception, static priors, whole-body IK and motion planning |
| `frozen_rgbd` | 5 | Unchanged controllers with source fingerprints and separate dependency pins |
| `larchenko_ckpt2` / `larchenko_ckpt3` | 11 / 3 | Adapted 2025 π0.5 policies |
| `comet2` / `comet2_1` / `comet3` | 7 / 6 / 33 | Team π0.5 policies; `comet3` also handles the fallback `*` route |

All **100 tasks** have a route. Runtime observations are camera images, depth where used, robot-relative camera poses and proprioception; geometric controllers also include static robot, scene and object priors. VLA backends use RGB and proprioception. The COMET runtime replans after **20 executed actions**, with its bundled scheduler preserving action compression and gripper behavior. Each evaluator must have its own policy server.

## Run the released policy

Use Linux x86-64, Docker with NVIDIA Container Toolkit, an NVIDIA GPU with **24 GB VRAM for the policy**, and the separately installed [BEHAVIOR-1K v3.9.3-post1 evaluator](https://github.com/StanfordVL/BEHAVIOR-1K/tree/v3.9.3-post1) and its assets. The simulator needs additional resources: policy VRAM is not a total system memory budget. The public image is approximately **71.1 GiB compressed**; allow additional space for unpacking and extracted checkpoints.

```bash
git clone https://github.com/wingrune/VERA-B1K-2026.git
cd VERA-B1K-2026

IMAGE=ghcr.io/alstar8/b1k-evo-submit@sha256:1ab7d94f5b3ecc347dbb9adccc2d9212cfbd2ed4aad078bda82faad607a240c5
docker pull "$IMAGE"
docker run -d --rm --gpus device=0 --name vera-policy -p 8000:8000 "$IMAGE"
curl --fail --retry 30 --retry-connrefused --retry-delay 2 http://127.0.0.1:8000/healthz
```

The digest identifies the released image exactly. It contains the model weights and inference dependencies and requires no policy network access at runtime. `/healthz` checks the server; model loading and compilation happen when a task is first requested.

With the evaluator's Python environment activated, run one radio episode from this repository root:

```bash
OMNI_KIT_ACCEPT_EULA=YES python -m omnigibson.eval.eval \
  --task-name turning_on_radio --mode public_test \
  --host 127.0.0.1 --port 8000 \
  --robot-config "$PWD/eval/r1pro.yaml" \
  --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
  --replay-action-chunk-size 1 \
  --instance-indices 0 --num-envs 1 --num-rollouts 1 \
  --output-dir "$PWD/results/turning_on_radio" --write-video

python3 scripts/check_results.py results/turning_on_radio --metrics-only --strict
docker stop vera-policy
```

Index `0` is public instance `301`. For the ten-instance validation, use indices `0` through `9`, one evaluator process per instance, default timeouts and one rollout. Larchenko routes use **DefaultWrapper**; COMET routes use **RGBDFullResWrapper** and **20-action replay**. Follow the [complete evaluation commands](docs/reproduce.md) for automatic wrapper selection and all routed tasks.

## Rebuild from source

First recover the released weights without starting a policy container (a GPU is not needed for extraction):

```bash
python3 scripts/extract_release_checkpoints.py
python3 server/routing.py config/routing.csv
python3 scripts/build_docker.py --tag vera-b1k:source
docker run --rm --gpus all -p 8000:8000 vera-b1k:source
```

Extraction refuses to overwrite existing checkpoints. It reads their paths from the image, restores the source layout and verifies `comet2` against the included checksum manifest. Separate public checkpoint repositories for the team's three COMET models were not present in the public VERA-VLA listing when this release was prepared; the pinned image is the available distribution. The [weights guide](docs/weights.md) also documents the upstream Larchenko download and remaining verification limits.

For a **code-only check**, without weights or GPU inference:

```bash
python3 tools/verify_source_snapshot.py
python3 server/routing.py --no-files
python3 tools/check_frozen_rgbd.py
python3 scripts/build_docker.py --dry-run --no-files --tag vera-b1k:source
```

The last command prepares and prints a build plan; it does not build an image. Source rebuilding depends on the external package registries named in the Dockerfiles. Use the prebuilt digest to recover the original packaged environment.

## Public validation evidence

The pinned [Hugging Face snapshot](https://huggingface.co/datasets/VERA-VLA/b1k_final_submission/tree/a7e21a893581e5f6a1bf71d8a0e9780530ecf241/standard.public.VERA_team.team.20261015) contains **820 JSON episodes across 82 tasks**, each with a corresponding video entry: instances **301–310**, rollout **0**. There are **259 successful episodes**. Mean final Q across those 82 published tasks is **0.531441**.

These describe the published ten-instance slice, **not an official leaderboard score**. Eighteen tasks have no episodes in this snapshot. The current routing configuration is not per-episode provenance: some task artifacts were produced by earlier specialists. Per-task README files and `solution.json` records distinguish native evaluator measurements from historical H100/Isaac Lab evolution runs. The blog's results explorer shows all 100 tasks and marks missing evidence explicitly.

| Task | Successes / episodes | Mean final Q |
|---|---:|---:|
| Installing smoke detectors | 10 / 10 | 1.000 |
| Installing a fax machine | 10 / 10 | 1.000 |
| Turning on radio | 8 / 10 | 0.800 |
| Collecting aluminum cans | 3 / 10 | 0.850 |
| Attaching a camera to a tripod | 6 / 10 | 0.600 |
| Making rose centerpieces | 0 / 10 | 0.400 |

Recompute the release summary without downloading videos:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/VERA-VLA/b1k_final_submission /tmp/vera-validation
git -C /tmp/vera-validation checkout a7e21a893581e5f6a1bf71d8a0e9780530ecf241
python3 tools/summarize_public_results.py /tmp/vera-validation
```

The summary tool reads committed JSON blobs and records their SHA256 hashes. Video matching verifies dataset file presence; it does not decode every video.

## Repository map

- `server/`, `config/`: evaluator protocol, backend lifecycle, task IDs and routing.
- `tasks/<task>/`: controller/planner code, priors, task notes and provenance.
- `solutions/`: backend adapters, bundled VLA runtime and third-party notices.
- `docker/`, `scripts/`: inference environments, image builder and evaluation utilities.
- `eval/`: bundled evaluator wrapper and R1Pro configuration copies.
- `checkpoints/`, `results/`: local weights and results, excluded from Git.
- `docs/`: GitHub Pages blog, technical report, reproduction and results data.
- `tests/`, `tools/`, `release/`: CPU checks, evidence tools and release provenance.

## Tests and project website

CPU tests need NumPy, Pillow, msgpack and websockets. They do not run the simulator or real model inference:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy==2.4.6 pillow==11.2.1 msgpack==1.1.0 websockets==16.1.1
PYTHONPATH="$PWD/solutions/comet2/runtime" .venv/bin/python -m unittest discover -s tests -v
```

Four optional differential tests require `COMET2_REFERENCE_BACKEND` and saved observations. The release checks and [GitHub Pages publishing instructions](docs/publishing.md) are documented separately. The website is plain HTML, CSS and JavaScript in `docs/`, with no build dependency.

## License and acknowledgments

The inherited project license is **[Apache-2.0](LICENSE)**. Preserve the licenses and notices shipped with each third-party component. Model weights, tokenizer assets and simulator/dataset assets retain their respective terms; this repository's license does not relicense them. See [third-party acknowledgments](THIRD_PARTY_NOTICES.md).

This release brings together the VERA team's task-specific planners and classical controllers, the adapted [Ilia Larchenko BEHAVIOR solution](https://github.com/IliaLarchenko/behavior-1k-solution), [OpenPI](https://github.com/Physical-Intelligence/openpi), and [cuRobo](https://github.com/NVlabs/curobo), evaluated with [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K). Historical development instructions are retained in [the source README archive](docs/legacy-development-notes.md).
