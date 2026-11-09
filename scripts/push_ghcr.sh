#!/bin/bash
# Push the submission image to the GitHub container registry (what the challenge portal's "Docker image URI" field takes).
#
# usage: scripts/push_ghcr.sh <local tag> [<ghcr repository>]     e.g. scripts/push_ghcr.sh b1k-evo-submit:submit-940bead
#   default repository ghcr.io/alstar8/b1k-evo-submit; pushed as :<local tag's tag> and :latest
# GHCR_TOKEN (classic GitHub token with write:packages) and GHCR_USER (default alstar8) come from .env unless set in the
# environment. The login is removed after the push (no credential store here: docker would keep the token in plain text).
# A new package is private: make it public once in GitHub > Packages > b1k-evo-submit > Package settings.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/scripts/env.sh"
load_env "$ROOT/.env"
[ $# -ge 1 ] || { sed -n 2,9p "$0"; exit 2; }
: "${GHCR_TOKEN:?set GHCR_TOKEN in .env (classic token with write:packages)}"
IMAGE=$1
DEST=${2:-ghcr.io/alstar8/b1k-evo-submit}
TAG=${IMAGE##*:}

docker image inspect "$IMAGE" >/dev/null
# GHCR cancels an upload after 10 minutes. docker pushes 5 layers at once by default, which splits the bandwidth until the 2-5 GB
# layers time out; one at a time each gets the full link. Wait until the daemon runs with "max-concurrent-uploads": 1.
uploads_ok() {
  python3 -c 'import json,sys; sys.exit(json.load(open("/etc/docker/daemon.json")).get("max-concurrent-uploads") != 1)' 2>/dev/null || return 1
  started=$(date -d "$(ps -o lstart= -C dockerd | head -1)" +%s); [ "$started" -gt "$(stat -c %Y /etc/docker/daemon.json)" ]
}
if ! uploads_ok; then
  echo "$(date -Is) WAITING: set \"max-concurrent-uploads\": 1 in /etc/docker/daemon.json and restart docker:"
  echo "  sudo python3 -c 'import json;p=\"/etc/docker/daemon.json\";d=json.load(open(p));d[\"max-concurrent-uploads\"]=1;d[\"max-upload-attempts\"]=10;json.dump(d,open(p,\"w\"),indent=4)' && sudo systemctl restart docker"
  until uploads_ok && docker info >/dev/null 2>&1; do sleep 20; done
  echo "$(date -Is) docker restarted with sequential uploads"
fi
echo "$GHCR_TOKEN" | docker login ghcr.io -u "${GHCR_USER:-alstar8}" --password-stdin
trap 'docker logout ghcr.io >/dev/null' EXIT
for t in "$TAG" latest; do
  docker tag "$IMAGE" "$DEST:$t"
  docker push "$DEST:$t"
done
digest=$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$IMAGE" | grep "^$DEST@" | head -1)
echo "pushed $DEST:$TAG and $DEST:latest"
echo "portal: Docker image URI $DEST:$TAG    digest ${digest#*@}"
