#!/usr/bin/env bash
set -euo pipefail

# Run a command with Colima available.
# Starts Colima if needed and stops it on exit only if this script started it.
# Usage: with_colima.sh <command> [args...]

if [[ $# -lt 1 ]]; then
  echo "Usage: with_colima.sh <command> [args...]" >&2
  exit 2
fi

command -v colima >/dev/null || { echo "Colima not found" >&2; exit 1; }

started_by_script=0

if ! colima status >/dev/null 2>&1; then
  colima start
  started_by_script=1
fi

cleanup() {
  if [[ "${started_by_script}" -eq 1 ]]; then
    colima stop
  fi
}
trap cleanup EXIT

"$@"
