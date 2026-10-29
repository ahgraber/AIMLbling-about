#!/usr/bin/env bash
set -euo pipefail

# Run the local Hugo container with an HTTP-friendly nginx config, using the selected
# container CLI (see scripts/container_engine.sh).
# Usage: hugo_run_local.sh <hugo_dir> [host_port]
#
# The image is nginx-unprivileged running as a non-root user, so nginx cannot
# bind the privileged port 80 inside the container without the NET_BIND_SERVICE
# capability that the k8s deployment grants but a local run does not. For local
# use the ephemeral config listens on 8080 (the unprivileged image's natural
# port), and the chosen host port (default 8081) maps to it. The config is a
# single-file `-v` mount: every runtime accepts that form, but Apple's runtime
# rejects a file source in `--mount type=bind`.

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: hugo_run_local.sh <hugo_dir> [host_port]" >&2
  exit 2
fi

hugo_dir="$1"
host_port="${2:-8081}"
image="ghcr.io/ahgraber/aimlbling-about:debug"

# Load .env as the build script does, so AIMLBLING_ENGINE selects the same CLI.
if [[ -f "${hugo_dir}/../.env" ]]; then
  set -a
  # shellcheck disable=SC1090,SC1091
  source "${hugo_dir}/../.env"
  set +a
fi

# shellcheck source=container_engine.sh disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/container_engine.sh"
engine_load

local_conf="$(mktemp "${hugo_dir}/docker/default.local.conf.XXXXXX")"

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
engine_run --rm -p "${host_port}:8080" -v "${local_conf}:/etc/nginx/conf.d/default.conf:ro" "${image}"
