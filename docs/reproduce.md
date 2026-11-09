# Reproduce VERA-Volve

Use the released image for the packaged inference environment. Use a source build to inspect or modify the approach. Public validation artifacts are a separate immutable reference, so a new experiment can be compared without overwriting the recorded outputs.

## Environment

- Linux x86-64 and Docker with NVIDIA Container Toolkit.
- A compatible NVIDIA GPU; the submission describes 24 GB VRAM for one active policy backend. Account separately for simulator GPU memory and host RAM.
- [BEHAVIOR-1K v3.9.3-post1](https://github.com/StanfordVL/BEHAVIOR-1K/tree/v3.9.3-post1), its supported OmniGibson/Isaac Sim environment and licensed assets. Follow the upstream installation instructions for this version.
- About 71.1 GiB of image download plus space for unpacking; source builds and checkpoint extraction need additional storage.

The policy and evaluator are separate processes. The Docker image contains policy dependencies; the evaluator runs under its own installed Python environment. No GitHub or Hugging Face token is needed to read the public release artifacts.

## Start the policy

```bash
IMAGE=ghcr.io/alstar8/b1k-evo-submit@sha256:1ab7d94f5b3ecc347dbb9adccc2d9212cfbd2ed4aad078bda82faad607a240c5
docker pull "$IMAGE"
docker run -d --rm --gpus device=0 --name vera-policy -p 8000:8000 "$IMAGE"
curl --fail --retry 30 --retry-connrefused --retry-delay 2 http://127.0.0.1:8000/healthz
```

The server routes by `obs['task_id']`. It loads one backend at a time; the first policy call can take time for imports, model loading, IK initialization or compilation. A health response confirms the server, not completion of GPU warmup. Use `docker logs vera-policy` for startup and task-routing logs.

## Evaluate a chosen set or all tasks

Activate the evaluator's Python environment and run this block from the VERA repository root. It obtains routing from the running image, chooses the wrapper and replay size for each backend, and starts a fresh native evaluator process for every instance. To start with one task, filter the generated task list as shown below before running the loop.

```bash
set -euo pipefail
mkdir -p .build

docker cp vera-policy:/opt/submit/config/routing.csv .build/release-routing.csv
docker cp vera-policy:/opt/submit/config/solutions.json .build/release-solutions.json
python - <<'PY' > .build/evaluation-tasks.tsv
import csv, json
solutions = json.load(open('.build/release-solutions.json'))
for row in csv.DictReader(open('.build/release-routing.csv')):
    task, solution = row['task'], row['solution']
    if task == '*' or solution in ('', 'none'):
        continue
    wrapper = 'DefaultWrapper' if solution.startswith('larchenko_ckpt') else 'RGBDFullResWrapper'
    chunk = solutions[solution].get('action_chunk_size', 1)
    print(task, wrapper, chunk, sep='\t')
PY

MODE=public_test
POLICY_PORT=8000
OUTPUT="$PWD/results/reproduced"
while IFS=$'\t' read -r TASK WRAPPER CHUNK; do
  for INDEX in {0..9}; do
    mkdir -p "$OUTPUT/$TASK/logs"
    OMNI_KIT_ACCEPT_EULA=YES python -m omnigibson.eval.eval \
      --task-name "$TASK" --mode "$MODE" \
      --host 127.0.0.1 --port "$POLICY_PORT" \
      --robot-config "$PWD/eval/r1pro.yaml" \
      --env-wrapper "omnigibson.eval.wrappers.$WRAPPER" \
      --replay-action-chunk-size "$CHUNK" \
      --instance-indices "$INDEX" --num-envs 1 --num-rollouts 1 \
      --output-dir "$OUTPUT/$TASK" --write-video < /dev/null \
      > "$OUTPUT/$TASK/logs/${TASK}_$((301 + INDEX)).log" 2>&1 \
      || echo "Evaluator failed: $TASK index $INDEX; see logs" >&2
  done
done < .build/evaluation-tasks.tsv
```

For a single task, filter the task list before the loop:

```bash
awk -F '\t' '$1 == "turning_on_radio"' .build/evaluation-tasks.tsv > .build/one-task.tsv
# Replace the final loop input with: done < .build/one-task.tsv
```

Indices 0–9 correspond to instances 301–310. Keep the evaluator's default horizons, `--num-envs 1`, one rollout and `--write-video`. Use one policy container per concurrent evaluator, with distinct ports, runtime/cache locations and output directories. Afterward:

```bash
python3 scripts/check_results.py results/reproduced/turning_on_radio --metrics-only --strict
docker stop vera-policy
```

The original `scripts/run_eval.sh` remains unchanged for source development. It defaults to RGBDFullResWrapper for every route; use the wrapper selection above for Larchenko routes when matching the released submission instructions. Its default indices are the same ten-instance slice. `check_results.py` also prints an aggregate with its own public-instance denominator; do not confuse that output with the observed ten-instance mean or a verified leaderboard score.

## Rebuild the source

```bash
python3 tools/verify_source_snapshot.py
python3 scripts/extract_release_checkpoints.py
python3 server/routing.py config/routing.csv
python3 scripts/build_docker.py --tag vera-b1k:source
```

See the [weights guide](weights.md). The root Dockerfile pins cuRobo and planner dependencies; frozen RGB-D, Larchenko and COMET run in separate environments. The builder includes each needed runtime and checkpoint according to the routing table. A source rebuild may resolve some external installer/base-image references again; the pinned prebuilt image preserves the packaged environment.

A selected planner can be evaluated without downloading VLA weights using the existing task-scoped launcher:

```bash
B1K=/absolute/path/to/BEHAVIOR-1K \
B1K_PYTHON=/absolute/path/to/evaluator/python \
INDICES=0 RESULTS_DIR=results_smoke \
bash scripts/run_eval.sh turning_on_radio
```

This builds and starts its policy image and runs instance 301. Before a new full validation, test the exact changed planner image with the native evaluator and validate the JSON/video output. H100/Isaac Lab evolution results do not substitute for a native evaluator run.

## Check the released evidence

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/VERA-VLA/b1k_final_submission /tmp/vera-validation
git -C /tmp/vera-validation checkout a7e21a893581e5f6a1bf71d8a0e9780530ecf241
python3 tools/summarize_public_results.py /tmp/vera-validation
```

This reads committed JSON blobs, checks task/instance identity, completeness of the ten-instance task sets, Q ranges, success consistency and corresponding video paths, then writes `docs/data/public-results.json` and `docs/data/evidence-manifest.json`. It does not fetch or decode all MP4 files. Use Hugging Face's normal download tools to obtain the actual videos if needed; a Git clone with `GIT_LFS_SKIP_SMUDGE=1` leaves media pointers.

The snapshot contains 82 tasks and 820 episodes. Mean Q over published tasks is 0.531441, with 259 episodes reporting success. Missing task evidence and current routes are both displayed in the project blog. The release route is not proof of which backend generated an older episode.

## CPU verification

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy==2.4.6 pillow==11.2.1 msgpack==1.1.0 websockets==16.1.1
PYTHONPATH="$PWD/solutions/comet2/runtime" .venv/bin/python -m unittest discover -s tests -v
python3 tools/verify_source_snapshot.py
python3 tools/check_frozen_rgbd.py
python3 server/routing.py --no-files
python3 scripts/build_docker.py --dry-run --no-files
```

The bundled client path is needed by the transport tests. Four optional frozen-reference differential tests are skipped without external fixtures. CPU tests cover routing, action transport, backend recovery, source fingerprints and build planning; they do not measure real policy success.
