#!/bin/bash
# Test the code currently in this repository on one or more tasks, results into results_tmp/<task>/ (never into results/).
#
# usage: scripts/test_task.sh <task id or name> [...]          e.g. scripts/test_task.sh 83 attach_a_camera_to_a_tripod
#   env: everything scripts/run_eval.sh reads (B1K, B1K_PYTHON, INDICES), plus
#        PORT       host port of the test container (default 8000)
#        TAG        image tag (default b1k-evo-submit:test); NO_BUILD=1 reuses the existing image
#        ROUTING    routing table to build with (default config/routing.csv; each tested task must be routed in it)
# Builds the image from the working tree, starts it, runs the official evaluator for the tasks (instances 301-310) into
# results_tmp/<task>/{json,videos,logs}, stops the container and checks each task folder. Compare and move a task into
# results/ with scripts/promote_task.py <task>.
set -euo pipefail
[ $# -ge 1 ] || { sed -n 2,12p "$0"; exit 2; }
REPO=$(cd "$(dirname "$0")/.." && pwd)
PORT=${PORT:-8000}
TAG=${TAG:-b1k-evo-submit:test}
ROUTING=${ROUTING:-$REPO/config/routing.csv}

tasks=()
for t in "$@"; do
  tasks+=("$(python3 -c "
import json, sys
ids = json.load(open('$REPO/server/task_ids_2026.json'))
t = sys.argv[1]
name = {str(v): k for k, v in ids.items()}.get(t, t)
sys.exit(f'unknown task {t}') if name not in ids else print(name)" "$t")")
done
# build only what the tested tasks need: their rows of the routing table
mkdir -p "$REPO/.build"
TEST_ROUTING=$REPO/.build/routing_test.csv
python3 - "$ROUTING" "$TEST_ROUTING" "${tasks[@]}" <<'PY'
import csv, sys
src, dst, tasks = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = [r for r in csv.DictReader(l for l in open(src) if l.strip() and not l.lstrip().startswith('#'))]
keep = [r for r in rows if r['task'] in tasks and r['solution'] not in ('', 'none')]
missing = sorted(set(tasks) - {r['task'] for r in keep})
if missing:
    sys.exit(f'not routed in {src}: {missing}')
with open(dst, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['task', 'solution', 'checkpoint'])
    w.writerows([r['task'], r['solution'], r['checkpoint']] for r in keep)
print('routes:', {r['task']: r['solution'] for r in keep})
PY

[ "${NO_BUILD:-0}" = 1 ] || python3 "$REPO/scripts/build_docker.py" --routing "$TEST_ROUTING" --tag "$TAG"
name=b1k-test-$$
docker run -d --gpus all --name "$name" -p "$PORT:8000" "$TAG" >/dev/null
trap 'for t in "${tasks[@]}"; do mkdir -p "$REPO/results_tmp/$t/logs"; docker logs "$name" > "$REPO/results_tmp/$t/logs/container_$name.log" 2>&1 || true; done; docker rm -f "$name" >/dev/null' EXIT
until curl -sf "localhost:$PORT/healthz" >/dev/null; do sleep 1; done

RESULTS_DIR=results_tmp IMAGE=$TAG PORT=$PORT B1K_EVALUATOR_CONTAINER=1 "$REPO/scripts/run_eval.sh" "${tasks[@]}" || true
echo "compare / promote: python3 scripts/promote_task.py <task>"
