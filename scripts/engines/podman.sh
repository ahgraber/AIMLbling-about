# shellcheck shell=bash
# Podman backend. See scripts/container_engine.sh for the function contract.
#
# Builds pass BUILDARCH explicitly as the podman host's architecture, which runs the
# Dockerfile's $BUILDPLATFORM stage. Cross-arch builds use the podman machine's
# qemu-user-static emulation where a RUN step needs it.

_podman_build_arch() {
  local arch
  arch="$(podman info --format '{{.Host.Arch}}')"
  case "${arch}" in
    amd64 | x86_64) echo amd64 ;;
    arm64 | aarch64) echo arm64 ;;
    *)
      echo "Unsupported podman build architecture: ${arch}" >&2
      return 2
      ;;
  esac
}

# A manifest from engine_push and a plain image from engine_build share IMAGE's name.
# Podman appends to an existing manifest and cannot replace a same-named image of the
# other kind, so delete whichever exists first.
_podman_clear() {
  local image="$1"
  if podman manifest exists "${image}"; then
    podman manifest rm "${image}" > /dev/null
  elif podman image exists "${image}"; then
    podman image rm "${image}" > /dev/null
  fi
}

engine_build() {
  local image="$1" platform="$2" arch
  shift 2
  arch="$(_podman_build_arch)"
  _podman_clear "${image}"
  podman build --format docker ${platform:+--platform "${platform}"} \
    --build-arg "BUILDARCH=${arch}" --tag "${image}" "$@"
}

engine_push() {
  local image="$1" platforms="$2" arch
  shift 2
  arch="$(_podman_build_arch)"
  _podman_clear "${image}"
  podman build --format docker --platform "${platforms}" \
    --build-arg "BUILDARCH=${arch}" --manifest "${image}" "$@"
  podman manifest push --all "${image}" "docker://${image}"
}

engine_run() {
  podman run "$@"
}
