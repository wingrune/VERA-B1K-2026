#!/bin/bash
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: scripts/run_eval.sh <task> [<task> ...]"; exit 2; }
REPO=$(cd "$(dirname "$0")/.." && pwd)
B1K=${B1K:-$HOME/BEHAVIOR-1K}
PY=${B1K_PYTHON:-python}
INDICES=${INDICES:-0 1 2 3 4 5 6 7 8 9}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
IMAGE=${IMAGE:-b1k-evo-submit:latest}
[ -d "$B1K/OmniGibson" ] || { echo "no BEHAVIOR-1K checkout at B1K=$B1K (set B1K)" >&2; exit 1; }
command -v "$PY" >/dev/null || { echo "B1K_PYTHON=$PY not found (set it to the python of the OmniGibson env)" >&2; exit 1; }

if [ "${B1K_EVALUATOR_CONTAINER:-0}" != 1 ]; then
  command -v docker >/dev/null || { echo "docker not found; install Docker with GPU support" >&2; exit 1; }
  ROUTING=${ROUTING:-$REPO/config/routing.csv}
  TAG=${TAG:-b1k-evo-submit:$(date +%Y%m%d-%H%M%S)}
  ROUTED=$REPO/.build/routing_eval_$$.csv
  mkdir -p "$REPO/.build"
  python3 - "$ROUTING" "$ROUTED" "$REPO/server/task_ids_2026.json" "$@" <<'PY'
import csv, json, sys
source, target, ids_path, *requested = sys.argv[1:]
ids = json.load(open(ids_path))
by_number = {str(number): name for name, number in ids.items()}
tasks = [by_number.get(task, task) for task in requested]
unknown = sorted(set(tasks) - set(ids))
if unknown:
    sys.exit(f"unknown 2026 task(s): {', '.join(unknown)}")
with open(source, newline='') as f:
    rows = list(csv.DictReader(line for line in f if line.strip() and not line.lstrip().startswith('#')))
selected = [row for row in rows if row['task'] in tasks and row['solution'] not in ('', 'none')]
missing = sorted(set(tasks) - {row['task'] for row in selected})
if missing:
    sys.exit(f"task(s) have no planner route in {source}: {', '.join(missing)}")
with open(target, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['task', 'solution', 'checkpoint'])
    w.writerows((row['task'], row['solution'], row['checkpoint']) for row in selected)
print('Docker routes:', ', '.join(f"{r['task']}={r['solution']}" for r in selected))
PY
  python3 "$REPO/scripts/build_docker.py" --routing "$ROUTED" --tag "$TAG"
  rm -f "$ROUTED"
  NAME=b1k-eval-$$
  docker run -d --gpus all --name "$NAME" -p "$PORT:8000" "$TAG" >/dev/null
  TASKS=("$@")
  cleanup() {
    for t in "${TASKS[@]}"; do
      mkdir -p "$REPO/${RESULTS_DIR:-results}/$t/logs"
      docker logs "$NAME" > "$REPO/${RESULTS_DIR:-results}/$t/logs/container_$NAME.log" 2>&1 || true
    done
    docker rm -f "$NAME" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT
  for _ in $(seq 1 300); do
    if curl -sf "http://$HOST:$PORT/healthz" >/dev/null; then break; fi
    sleep 1
  done
  curl -sf "http://$HOST:$PORT/healthz" >/dev/null || { echo "policy container did not become healthy; see docker logs $NAME" >&2; exit 1; }
  B1K_EVALUATOR_CONTAINER=1 IMAGE="$TAG" HOST="$HOST" PORT="$PORT" "$0" "$@"
  exit $?
fi

WRAPPER=omnigibson.eval.wrappers.RGBDFullResWrapper
ROOT_OUT=$REPO/${RESULTS_DIR:-results}
status=0

for task in "$@"; do
  CHUNK_SIZE=$(python3 "$REPO/scripts/eval_policy_args.py" "$task" --routing "${ROUTING:-$REPO/config/routing.csv}")
  replay_args=()
  if [ "$CHUNK_SIZE" -gt 1 ]; then
    replay_args+=(--replay-action-chunk-size "$CHUNK_SIZE")
  fi
  OUT=$ROOT_OUT/$task
  mkdir -p "$OUT/json" "$OUT/videos" "$OUT/logs"
  cp "$REPO/eval/r1pro.yaml" "$REPO/eval/rgbd_full_res_wrapper.py" "$OUT/"
  for i in $INDICES; do
    inst=$((301 + i))
    if [ -f "$OUT/json/${task}_${inst}_0.json" ]; then echo "skip $task $inst (done)"; continue; fi
    echo "$(date -Is) $task index $i (instance $inst)"
    (cd "$B1K" && OMNI_KIT_ACCEPT_EULA=YES "$PY" -m omnigibson.eval.eval --task-name "$task" --host "$HOST" --port "$PORT" \
        --instance-indices "$i" --num-envs 1 --num-rollouts 1 --env-wrapper "$WRAPPER" \
        --output-dir "$OUT" --write-video ${replay_args[@]+"${replay_args[@]}"}) > "$OUT/logs/${task}_${inst}.log" 2>&1 \
      || echo "evaluator failed for $task $inst, see $OUT/logs/${task}_${inst}.log"
  done
  cat > "$OUT/README.md" <<EOF

Policy: Docker image \`$IMAGE\` (repository b1k_evo_submit). It serves one websocket policy on port 8000 and routes this task
(\`obs["task_id"]\`) to its solution in \`config/routing.csv\`.

Start the policy (one GPU, 24 GB):

    docker run --gpus all --rm -p 8000:8000 $IMAGE

Evaluator (BEHAVIOR-1K v3.9.3-post1, from the repository root), one process per instance:

    python -m omnigibson.eval.eval --task-name $task --host <host> --port 8000 --instance-indices <i> --num-envs 1 \\
        --num-rollouts 1 --env-wrapper $WRAPPER --output-dir <out> --write-video ${replay_args[*]-}

- Wrapper: \`$WRAPPER\` (unchanged, \`rgbd_full_res_wrapper.py\` in this folder): RGB + depth + proprioception only.
- Robot config: the bundled \`omnigibson/eval/r1pro.yaml\`, unchanged (no \`--robot-config\`); a copy is in this folder.
- Default timeouts (1.5x mean human length), one rollout per instance, instances 301-310 (indices 0-9).
- \`--num-envs 1\` is required: the policy keeps one episode state.
- Action replay chunk size: $CHUNK_SIZE (from the routed solution's configuration).
- Logs for this task are in \`logs/\`.
EOF
  python3 "$REPO/scripts/check_results.py" "$OUT" || status=1
done
exit "$status"
