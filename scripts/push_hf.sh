#!/bin/bash
# Push results (json, videos, evaluator logs, package files) and the submission Docker image to a Hugging Face dataset repo.
#
# usage: scripts/push_hf.sh [--repo <hf repo id>] [--image <docker tag> [--replace]] [results/<task> ...]
#   example: scripts/push_hf.sh --image b1k-evo-submit:latest results/store_honey
#   a parent directory expands to each results/<task> inside it: scripts/push_hf.sh results
# HF_TOKEN and HF_REPO come from .env in the repository root (gitignored, see .env.example) unless already set in the
# environment; --repo overrides HF_REPO. The repo is created private if missing (dataset type). Each task folder goes to
# results/<task>/; the
# image is saved with `docker save`, gzipped and uploaded to docker/<tag>.tar.gz (20 GB parts docker/<tag>.tar.gz.part-NN when
# over 45 GB: Hugging Face's file limit is 50 GB) with docker/README.md (sha256, load and run commands). --replace then removes
# the other images under docker/ (they stay in the repo history). Large archives: set TMPDIR to a disk with room (/data).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/scripts/env.sh"
load_env "$ROOT/.env"
[ $# -ge 1 ] || { sed -n 2,9p "$0"; exit 2; }
: "${HF_TOKEN:?set HF_TOKEN in .env (cp .env.example .env)}"
export HF_TOKEN
command -v hf >/dev/null || { echo "hf CLI missing: pip install -U 'huggingface_hub>=1.0'"; exit 2; }
REPO_ID=${HF_REPO:-}
IMAGE=""
REPLACE=0
RUNS=()
while [ $# -gt 0 ]; do
  case $1 in
    --repo) REPO_ID=$2; shift 2 ;;
    --image) IMAGE=$2; shift 2 ;;
    --replace) REPLACE=1; shift ;;
    *) RUNS+=("$1"); shift ;;
  esac
done
[ -n "$REPO_ID" ] || { echo "no Hugging Face repo: set HF_REPO in .env or pass --repo"; exit 2; }

hf repos create "$REPO_ID" --repo-type dataset --private --exist-ok

expand() {
  python3 - "$1" "$ROOT/server/task_ids_2026.json" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
ids = set(json.loads(Path(sys.argv[2]).read_text()))
subs = [p for p in sorted(root.iterdir()) if p.is_dir() and p.name in ids and (p / 'json').is_dir()] if root.is_dir() and root.name not in ids else []
print('\n'.join(str(p) for p in subs) if subs else str(root))
PY
}
for run in "${RUNS[@]}"; do
  while IFS= read -r folder; do
    [ -n "$folder" ] || continue
    python3 "$ROOT/scripts/check_results.py" "$folder" || { echo "not uploading $folder: fix the errors above"; exit 1; }
    hf upload "$REPO_ID" "$folder" "results/$(basename "$folder")" --repo-type dataset --commit-message "results $(basename "$folder")"
  done < <(expand "${run%/}")
done

if [ -n "$IMAGE" ]; then
  name=$(echo "$IMAGE" | tr '/:' '__')
  tmp=$(mktemp -d "${TMPDIR:-/tmp}/b1k_image.XXXXXX")
  trap 'rm -rf "$tmp"' EXIT
  echo "saving $IMAGE (several GB, takes a while)"
  docker save "$IMAGE" | gzip -1 > "$tmp/$name.tar.gz"
  digest=$(docker image inspect --format '{{.Id}}' "$IMAGE")
  sha=$(sha256sum "$tmp/$name.tar.gz" | cut -d' ' -f1)
  mkdir "$tmp/docker"
  # Hugging Face stores files up to 50 GB: larger archives go up as 20 GB parts, joined again with cat.
  if [ "$(stat -c %s "$tmp/$name.tar.gz")" -gt $((45 * 1024 ** 3)) ]; then
    split -b 20G -d "$tmp/$name.tar.gz" "$tmp/docker/$name.tar.gz.part-"
    rm "$tmp/$name.tar.gz"
    load="cat docker/$name.tar.gz.part-* | gunzip | docker load"
    check="cat docker/$name.tar.gz.part-* | sha256sum      # $sha"
  else
    mv "$tmp/$name.tar.gz" "$tmp/docker/"
    load="gunzip -c docker/$name.tar.gz | docker load"
    check="sha256sum docker/$name.tar.gz      # $sha"
  fi
  cat > "$tmp/docker/README.md" <<EOF
# Submission image $IMAGE

Image id: $digest

    hf download $REPO_ID --include "docker/$name.tar.gz*" --repo-type dataset --local-dir .
    $check
    $load
    docker run --gpus all --rm -p 8000:8000 $IMAGE

The container serves the BEHAVIOR 2026 websocket policy on port 8000 (health check /healthz). Run the evaluator with
--env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper --num-envs 1.
EOF
  hf upload "$REPO_ID" "$tmp/docker" docker --repo-type dataset --commit-message "image $IMAGE"
  if [ "$REPLACE" = 1 ]; then      # the previous images stay in the repository history
    python3 - "$REPO_ID" "docker/$name.tar.gz" <<'PY'
import sys
from huggingface_hub import HfApi
repo, keep = sys.argv[1:]
api = HfApi()
old = [f for f in api.list_repo_files(repo, repo_type='dataset')
       if f.startswith('docker/') and '.tar.gz' in f and not f.startswith(keep)]
if old:
    api.delete_files(repo, old, repo_type='dataset', commit_message=f'replace image: remove {len(old)} old file(s)')
print('removed previous image files:', old or 'none')
PY
  fi
fi
echo "done: https://huggingface.co/datasets/$REPO_ID"
