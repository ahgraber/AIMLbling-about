#!/usr/bin/env bash
set -euo pipefail

# Build the Hugo site container image using Docker.
# Usage: hugo_build_image.sh <repo_root> <hugo_dir>

if [[ $# -ne 2 ]]; then
  echo "Usage: hugo_build_image.sh <repo_root> <hugo_dir>" >&2
  exit 2
fi

repo_root="$1"
hugo_dir="$2"

if [[ -f "${repo_root}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090,SC1091
  source "${repo_root}/.env"
  set +a
fi

TREADMILL_PAT="${TREADMILL_PAT:-}"

build_date=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
vcs_ref=$(git -C "${repo_root}" rev-parse --short HEAD)

build_cmd=(
  docker build
  --build-arg "BUILD_DATE=${build_date}"
  --build-arg "VCS_REF=${vcs_ref}"
  -t ghcr.io/ahgraber/aimlbling-about:debug
  -f "${hugo_dir}/docker/Dockerfile"
)

if [[ -n "${TREADMILL_PAT}" ]]; then
  build_cmd+=(--secret "id=treadmill_pat,env=TREADMILL_PAT")
fi

build_cmd+=("${repo_root}")
"${build_cmd[@]}"
