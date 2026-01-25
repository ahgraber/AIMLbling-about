#!/usr/bin/env bash
set -euo pipefail

# Publish a draft post by toggling draft and updating date.
# Usage: hugo_publish.sh <repo_root> <path>

if [[ $# -ne 2 ]]; then
  echo "Usage: hugo_publish.sh <repo_root> <path>" >&2
  exit 2
fi

repo_root="$1"
path="$2"

if [[ "${path}" = /* ]]; then
  target_path="${path}"
else
  target_path="${repo_root}/${path}"
fi

if [[ ! -f "${target_path}" ]]; then
  echo "File not found" >&2
  exit 1
fi

current_date=$(date +%F)
sed -i '/draft: true/c\\draft: false' "${target_path}"
sed -i "/date: .*/c\\\\date: ${current_date}" "${target_path}"

echo "Commit this change, then branch is ready for PR!"
