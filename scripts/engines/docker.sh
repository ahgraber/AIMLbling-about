# shellcheck shell=bash
# Docker CLI backend (buildx). See scripts/container_engine.sh for the function contract.
#
# buildx sets BUILDARCH and the other platform build args itself. A multi-platform image
# cannot be loaded into a store without Docker's containerd image store (socktainer and
# podman's Docker API have none), so engine_push uploads straight from BuildKit instead.

engine_build() {
  local image="$1" platform="$2"
  shift 2
  docker buildx build ${platform:+--platform "${platform}"} --load --tag "${image}" "$@"
}

engine_push() {
  local image="$1" platforms="$2"
  shift 2
  docker buildx build --platform "${platforms}" --push --tag "${image}" "$@"
}

engine_run() {
  docker run "$@"
}
