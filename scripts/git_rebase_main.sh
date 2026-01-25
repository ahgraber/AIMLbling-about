#!/usr/bin/env bash
set -euo pipefail

# Rebase the current branch onto origin/main with merge preservation.
# Usage: git_rebase_main.sh <repo_root>

if [[ $# -ne 1 ]]; then
  echo "Usage: git_rebase_main.sh <repo_root>" >&2
  exit 2
fi

repo_root="$1"

git -C "${repo_root}" rebase origin/main --rebase-merges \
  || { echo "Merge conflicts detected, exiting task. Use 'git rebase --continue'."; exit 1; }
