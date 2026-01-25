#!/usr/bin/env bash
set -euo pipefail

# Create a new Hugo draft post, set its title, and create a blog branch.
# Usage: hugo_new_post.sh <repo_root> <hugo_dir> <title>

if [[ $# -ne 3 ]]; then
  echo "Usage: hugo_new_post.sh <repo_root> <hugo_dir> <title>" >&2
  exit 2
fi

repo_root="$1"
hugo_dir="$2"
title="$3"

slug=$(printf '%s' "${title}" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')
post_path="${hugo_dir}/content/blog/${slug}/index.md"

git -C "${repo_root}" checkout -b "blog/${slug}" main
hugo new content "blog/${slug}/index.md" -s "${hugo_dir}"

sed -i "/title: \"\"/c\\title: '{{ title \"${title}\" }}'" "${post_path}"
