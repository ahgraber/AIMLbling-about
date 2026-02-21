#!/usr/bin/env bash
set -euo pipefail

# Run the local Hugo container with an HTTP-friendly nginx config.
# Usage: hugo_run_local.sh <hugo_dir>

if [[ $# -ne 1 ]]; then
  echo "Usage: hugo_run_local.sh <hugo_dir>" >&2
  exit 2
fi

hugo_dir="$1"
image="ghcr.io/ahgraber/aimlbling-about:debug"
local_conf="$(mktemp "${hugo_dir}/docker/default.local.XXXXXX.conf")"

cleanup() {
  rm -f "${local_conf}"
}
trap cleanup EXIT

awk '/if \(\$forwarded_proto != '\''https'\''\) \{/ { skip=1; next } skip && /^[[:space:]]*\}/ { skip=0; next } !skip { print }' \
  "${hugo_dir}/docker/default.conf" > "${local_conf}"

echo "Access site at http://127.0.0.1"
echo "Press Ctrl+C to stop the container."
docker run --rm -p 80:80 -v "${local_conf}:/etc/nginx/conf.d/default.conf:ro" "${image}"
