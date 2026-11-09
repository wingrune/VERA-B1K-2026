#!/bin/bash
# Download (or resume) the Larchenko checkpoints the routing table uses into checkpoints/larchenko, retrying until complete,
# with one status line: downloaded / total, current speed, ETA.
#
# usage: scripts/download_checkpoints.sh [checkpoint_2 checkpoint_3 ...]     (default: checkpoint_2 checkpoint_3)
#   HF_TOKEN from the environment or .env is used when set (higher Hub rate limits). Ctrl-C stops; run again to resume.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
DIR=$(readlink -f "$REPO/checkpoints/larchenko")
HF_REPO=IliaLarchenko/behavior_submission
CKPTS=("$@"); [ ${#CKPTS[@]} -eq 0 ] && CKPTS=(checkpoint_2 checkpoint_3)
LOG="$DIR/../hf_download.log"
mkdir -p "$DIR"

# Only one downloader per folder: stop any earlier run.
pkill -f hf_dl_loop.sh 2>/dev/null
pkill -f "bin/hf download $HF_REPO" 2>/dev/null

if [ -z "${HF_TOKEN:-}" ] && [ -f "$REPO/.env" ]; then
  HF_TOKEN=$(sed -n 's/^HF_TOKEN=//p' "$REPO/.env" | tail -1)
fi
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN && echo "using HF_TOKEN" || unset HF_TOKEN

TOTAL=0
for c in "${CKPTS[@]}"; do
  n=$(curl -s "https://huggingface.co/api/models/$HF_REPO/tree/main/$c?recursive=true" \
    | python3 -c 'import json,sys; print(sum(f.get("size",0) for f in json.load(sys.stdin) if f.get("type")=="file"))' 2>/dev/null)
  TOTAL=$((TOTAL + ${n:-0}))
done
INCLUDES=(); for c in "${CKPTS[@]}"; do INCLUDES+=(--include "$c/*"); done
echo "${CKPTS[*]} -> $DIR ($(awk -v t=$TOTAL 'BEGIN{printf "%.1f GB", t/1e9}')), hf log: $LOG"

(
  until hf download "$HF_REPO" "${INCLUDES[@]}" --local-dir "$DIR" >> "$LOG" 2>&1; do
    echo "$(date -Is) hf download failed, retrying in 30 s" >> "$LOG"; sleep 30
  done
) &
DL=$!
trap 'kill $DL 2>/dev/null; pkill -f "bin/hf download $HF_REPO"; echo; echo stopped; exit 130' INT TERM

size() { du -sb "$DIR" 2>/dev/null | cut -f1; }
prev=$(size); tprev=$(date +%s)
while kill -0 $DL 2>/dev/null; do
  sleep 5
  cur=$(size); now=$(date +%s)
  awk -v c=$cur -v p=$prev -v dt=$((now - tprev)) -v t=$TOTAL 'BEGIN{
    s = (c - p) / (dt > 0 ? dt : 1); eta = "--";
    if (s > 0 && t > c) { e = (t - c) / s; eta = sprintf("%dh%02dm", e / 3600, (e % 3600) / 60) }
    printf "\r%6.2f / %.2f GB (%5.1f%%)  %7.2f MB/s  ETA %s   ", c/1e9, t/1e9, (t ? 100*c/t : 0), s/1e6, eta }'
  prev=$cur; tprev=$now
done
echo; wait $DL; echo "done: $(du -sh "$DIR"/checkpoint_* | tr '\n' ' ')"
