#!/usr/bin/env bash
set -euo pipefail

# Build the Hugo site container image using Podman.
# Usage: hugo_build_image.sh <repo_root> <hugo_dir> [platforms]
#
# platforms: comma-separated os/arch list (default: linux/amd64,linux/arm64, matching CI).
#   More than one platform builds a multi-arch manifest (podman requires --manifest, not
#   --tag, for that); a single platform builds a plain tagged image. Cross-arch builds use
#   the podman machine's qemu-user-static emulation — for this Dockerfile that is cheap
#   (native $BUILDPLATFORM build stage, RUN-free runtime stage). Pass platforms="linux/arm64"
#   for a native-only image if you just want to `just hugo run` locally.

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: hugo_build_image.sh <repo_root> <hugo_dir> [platforms]" >&2
  exit 2
fi

repo_root="$1"
hugo_dir="$2"
platforms="${3:-linux/amd64,linux/arm64}"

if [[ -f "${repo_root}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090,SC1091
  source "${repo_root}/.env"
  set +a
fi

TREADMILL_PAT="${TREADMILL_PAT:-}"

build_date=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
vcs_ref=$(git -C "${repo_root}" rev-parse --short HEAD)
image="ghcr.io/ahgraber/aimlbling-about:debug"

build_cmd=(
  podman build
  --build-arg "BUILD_DATE=${build_date}"
  --build-arg "VCS_REF=${vcs_ref}"
  --platform "${platforms}"
  -f "${hugo_dir}/docker/Dockerfile"
)

# podman requires --manifest (not --tag) when more than one platform is requested.
if [[ "${platforms}" == *,* ]]; then
  build_cmd+=(--manifest "${image}")
else
  build_cmd+=(-t "${image}")
fi

if [[ -n "${TREADMILL_PAT}" ]]; then
  build_cmd+=(--secret "id=treadmill_pat,env=TREADMILL_PAT")
fi

build_cmd+=("${repo_root}")
"${build_cmd[@]}"
