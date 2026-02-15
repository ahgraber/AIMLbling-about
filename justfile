set quiet := true
set shell := ['bash', '-euo', 'pipefail', '-c']
set dotenv-load := true

root_dir := shell('git -C "' + justfile_directory() + '" rev-parse --show-toplevel')

mod hugo ".taskfiles/hugo/justfile"
mod jupyter ".taskfiles/jupyter/justfile"


[doc('List available recipes')]
default:
    just --list


[doc('Catch feature branch up to main')]
catchup:
    "{{root_dir}}/scripts/git_catchup_main.sh" "{{root_dir}}"


[doc('Git rebase on main')]
_rebase:
    "{{root_dir}}/scripts/git_rebase_main.sh" "{{root_dir}}"
