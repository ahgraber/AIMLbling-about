#!/usr/bin/env bash
set -euo pipefail

# Build the Hugo site container image, or build and push it, with the selected container
# CLI (see scripts/container_engine.sh).
# Usage: hugo_build_image.sh <repo_root> <hugo_dir> build [platform]
#        hugo_build_image.sh <repo_root> <hugo_dir> push [platforms]
#
# build: one platform into the local image store (default: the engine's native platform).
# push:  a comma-separated platform list (default: linux/amd64,linux/arm64, matching CI),
#        pushed to the image's registry. Log in to ghcr.io first.
#
# The Dockerfile pins its build stage to $BUILDPLATFORM and its runtime stage has no RUN
# steps, so a non-native platform needs no emulated execution.

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: hugo_build_image.sh <repo_root> <hugo_dir> <build|push> [platforms]" >&2
  exit 2
fi

repo_root="$1"
hugo_dir="$2"
mode="$3"
image="ghcr.io/ahgraber/aimlbling-about:debug"

case "${mode}" in
  build) platforms="${4:-}" ;;
  push) platforms="${4:-linux/amd64,linux/arm64}" ;;
  *)
    echo "Unknown mode '${mode}'; use build or push" >&2
    exit 2
    ;;
esac

if [[ "${platforms}" == platforms=* ]]; then
  echo "Platform override is positional; use: just hugo ${mode} ${platforms#platforms=}" >&2
  exit 2
fi
if [[ "${platforms}" == *, || "${platforms}" == ,* || "${platforms}" == *,,* ]]; then
  echo "Empty entry in platform list '${platforms}'" >&2
  exit 2
fi
if [[ -n "${platforms}" ]]; then
  IFS=',' read -r -a platform_list <<< "${platforms}"
  for platform in "${platform_list[@]}"; do
    if [[ "${platform}" != linux/amd64 && "${platform}" != linux/arm64 ]]; then
      echo "Unsupported platform '${platform}'; use linux/amd64, linux/arm64, or both" >&2
      exit 2
    fi
  done
  if [[ "${mode}" == build && "${#platform_list[@]}" -gt 1 ]]; then
    echo "build loads one platform into the local store; use 'just hugo push ${platforms}' for several" >&2
    exit 2
  fi
fi

if [[ -f "${repo_root}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090,SC1091
  source "${repo_root}/.env"
  set +a
fi

# shellcheck source=container_engine.sh disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/container_engine.sh"
engine_load

build_date=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
vcs_ref=$(git -C "${repo_root}" rev-parse --short HEAD)

build_args=(
  --build-arg "BUILD_DATE=${build_date}"
  --build-arg "VCS_REF=${vcs_ref}"
  -f "${hugo_dir}/docker/Dockerfile"
)

# The Dockerfile's secret mount is optional, so pass the secret only when it is set.
if [[ -n "${TREADMILL_PAT:-}" ]]; then
  build_args+=(--secret "id=treadmill_pat,env=TREADMILL_PAT")
fi

"engine_${mode}" "${image}" "${platforms}" "${build_args[@]}" "${repo_root}"
