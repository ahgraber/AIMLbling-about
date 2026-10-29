# shellcheck shell=bash
# Container CLI selection for the Hugo image scripts.
#
# The scripts drive a container CLI, not a runtime: which runtime a CLI reaches (Docker
# Engine, Apple `container` through socktainer, a podman machine) is host configuration.
# Backends differ only in command syntax.
#
# Source this file, then call `engine_load`. It sets ENGINE to the selected CLI name and
# sources scripts/engines/<ENGINE>.sh, which defines:
#
#   engine_build IMAGE PLATFORM ARGS...  Build IMAGE into the local image store. PLATFORM is
#                                        one os/arch, or empty for the engine's native platform.
#                                        A rebuild replaces IMAGE.
#   engine_push IMAGE PLATFORMS ARGS...  Build IMAGE for PLATFORMS (comma-separated os/arch list)
#                                        and push every platform to IMAGE's registry.
#   engine_run ARGS...                   Run a container. ARGS are engine-neutral `run` flags
#                                        (--rm, -p, -v src:dst[:ro]) followed by the image.
#
# For engine_build and engine_push, ARGS are engine-neutral build flags (--build-arg,
# --secret id=...,env=..., -f) followed by the build context.
#
# Each backend file is named after the CLI it drives. To add a backend, add
# scripts/engines/<cli>.sh and put <cli> in _engine_candidates.
# Set AIMLBLING_ENGINE=<cli> to skip auto-selection.

_engine_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/engines"

# Preference order. The first CLI on PATH wins.
_engine_candidates="docker podman"

engine_load() {
  local name="${AIMLBLING_ENGINE:-}"
  local candidate backend
  local backends=()

  if [[ -n "${name}" ]]; then
    if [[ ! -f "${_engine_dir}/${name}.sh" ]]; then
      for backend in "${_engine_dir}"/*.sh; do
        backend="${backend##*/}"
        backends+=("${backend%.sh}")
      done
      echo "Unknown AIMLBLING_ENGINE '${name}'. Backends: ${backends[*]}" >&2
      return 2
    fi
    if ! command -v "${name}" > /dev/null; then
      echo "AIMLBLING_ENGINE=${name}, but '${name}' is not on PATH" >&2
      return 1
    fi
  else
    for candidate in ${_engine_candidates}; do
      if command -v "${candidate}" > /dev/null; then
        name="${candidate}"
        break
      fi
    done
    if [[ -z "${name}" ]]; then
      echo "No container CLI found. Install one of: ${_engine_candidates}" >&2
      return 1
    fi
  fi

  ENGINE="${name}"
  # shellcheck source=/dev/null
  source "${_engine_dir}/${ENGINE}.sh"
  echo "Container CLI: ${ENGINE}" >&2
}
