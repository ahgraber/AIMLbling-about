#!/usr/bin/env bash
set -euo pipefail

# Build the Hugo site container image using Podman.
# Usage: hugo_build_image.sh <repo_root> <hugo_dir> [platforms]
#
# platforms: comma-separated os/arch list (default: linux/amd64,linux/arm64, matching CI).
#   Every build updates a manifest-backed local name, including single-platform builds, so
#   switching between native and multi-arch builds never collides with a plain image tag.
#   Cross-arch builds use the podman machine's qemu-user-static emulation — for this
#   Dockerfile that is cheap
#   (native $BUILDPLATFORM build stage, RUN-free runtime stage). Pass linux/arm64 as the
#   positional build argument (`just hugo build linux/arm64`) for a native-only image.

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: hugo_build_image.sh <repo_root> <hugo_dir> [platforms]" >&2
  exit 2
fi

repo_root="$1"
hugo_dir="$2"
platforms="${3:-linux/amd64,linux/arm64}"

if [[ "${platforms}" == platforms=* ]]; then
  echo "Platform override is positional; use: just hugo build ${platforms#platforms=}" >&2
  exit 2
fi

if [[ -f "${repo_root}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090,SC1091
  source "${repo_root}/.env"
  set +a
fi

TREADMILL_PAT="${TREADMILL_PAT:-}"

build_arch=$(podman info --format '{{.Host.Arch}}')
case "${build_arch}" in
  amd64 | x86_64) build_arch=amd64 ;;
  arm64 | aarch64) build_arch=arm64 ;;
  *)
    echo "Unsupported Podman build architecture: ${build_arch}" >&2
    exit 2
    ;;
esac

build_date=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
vcs_ref=$(git -C "${repo_root}" rev-parse --short HEAD)
image="ghcr.io/ahgraber/aimlbling-about:debug"

# Keep the concise local tag backed by exactly one manifest containing the requested
# platforms. Podman appends to manifests and cannot replace a same-named plain image.
if podman manifest exists "${image}"; then
  podman manifest rm "${image}" >/dev/null
elif podman image exists "${image}"; then
  podman image rm "${image}" >/dev/null
fi

build_cmd=(
  podman build
  --format docker
  --build-arg "BUILD_DATE=${build_date}"
  --build-arg "VCS_REF=${vcs_ref}"
  --build-arg "BUILDARCH=${build_arch}"
  --platform "${platforms}"
  -f "${hugo_dir}/docker/Dockerfile"
)

build_cmd+=(--manifest "${image}")

if [[ -n "${TREADMILL_PAT}" ]]; then
  build_cmd+=(--secret "id=treadmill_pat,env=TREADMILL_PAT")
fi

build_cmd+=("${repo_root}")
"${build_cmd[@]}"
