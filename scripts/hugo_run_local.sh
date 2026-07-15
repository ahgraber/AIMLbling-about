#!/usr/bin/env bash
set -euo pipefail

# Run the local Hugo container with an HTTP-friendly nginx config.
# Usage: hugo_run_local.sh <hugo_dir> [host_port]
#
# The image is nginx-unprivileged running as a non-root user, so nginx cannot
# bind the privileged port 80 inside the container without the NET_BIND_SERVICE
# capability that the k8s deployment grants but a local rootless-podman run does
# not. For local use the ephemeral config listens on 8080 (the unprivileged
# image's natural port), and the chosen host port (default 8080) maps to it.

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: hugo_run_local.sh <hugo_dir> [host_port]" >&2
  exit 2
fi

hugo_dir="$1"
host_port="${2:-8080}"
image="ghcr.io/ahgraber/aimlbling-about:debug"
local_conf="$(mktemp "${hugo_dir}/docker/default.local.XXXXXX.conf")"

cleanup() {
  rm -f "${local_conf}"
}
trap cleanup EXIT

# Strip the HTTP->HTTPS redirect (no TLS locally) and move the server off the
# privileged port 80 to 8080 so the non-root nginx user can bind it locally.
awk '/if \(\$forwarded_proto != '\''https'\''\) \{/ { skip=1; next } skip && /^[[:space:]]*\}/ { skip=0; next } skip { next } { gsub(/listen 80;/, "listen 8080;"); gsub(/listen \[::\]:80;/, "listen [::]:8080;"); print }' \
  "${hugo_dir}/docker/default.conf" > "${local_conf}"

echo "Access site at http://127.0.0.1:${host_port}"
echo "Press Ctrl+C to stop the container."
podman run --rm -p "${host_port}:8080" -v "${local_conf}:/etc/nginx/conf.d/default.conf:ro" "${image}"
