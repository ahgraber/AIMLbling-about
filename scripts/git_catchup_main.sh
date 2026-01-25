#!/usr/bin/env bash
set -euo pipefail

# Stash local changes, rebase onto main, and restore stashed work.
# Usage: git_catchup_main.sh <repo_root>

if [[ $# -ne 1 ]]; then
  echo "Usage: git_catchup_main.sh <repo_root>" >&2
  exit 2
fi

repo_root="$1"

stash_output="$(git -C "${repo_root}" stash)"
stash_created=1
if [[ "${stash_output}" == "No local changes to save" ]]; then
  stash_created=0
fi

cleanup() {
  if [[ ${stash_created} -eq 1 ]]; then
    git -C "${repo_root}" stash pop
  fi
}
trap cleanup EXIT

git -C "${repo_root}" fetch origin main
"${repo_root}"/scripts/git_rebase_main.sh "${repo_root}"
echo "git rebase successful"
printf "Caught up with main!  To complete, use '\033[0;36mgit push --force-with-lease\033[0m'\n"
