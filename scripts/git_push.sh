#!/bin/bash
# Push a branch of this repository to GitHub with GITHUB_TOKEN from .env (gitignored, see .env.example).
#
# usage: scripts/git_push.sh [branch] [extra git push args]      (default: the current branch, remote origin)
# The token is handed to git by a credential helper for this one command only: it is not written into .git/config or the
# remote URL. Refuses to push when .env, results or checkpoint files are tracked by git.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/scripts/env.sh"
load_env "$ROOT/.env"
: "${GITHUB_TOKEN:?set GITHUB_TOKEN in .env (cp .env.example .env)}"
export GITHUB_TOKEN
cd "$ROOT"

leaked=$(git ls-files | grep -E '^(\.env$|results/|results_tmp/|checkpoints/)' | grep -vE '^(results|checkpoints)/README\.md$' || true)
[ -z "$leaked" ] || { printf 'tracked files that must stay local:\n%s\nremove them: git rm --cached <file>\n' "$leaked"; exit 1; }

branch=${1:-$(git branch --show-current)}
shift || true
git -c credential.helper= \
    -c credential.helper='!f() { [ "$1" = get ] && echo username=x-access-token && echo "password=$GITHUB_TOKEN"; }; f' \
    push origin "$branch" "$@"
