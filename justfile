set shell := ['bash', '-euo', 'pipefail', '-c']

root_dir := `git -C "{{justfile_directory()}}" rev-parse --show-toplevel`

mod hugo ".taskfiles/hugo/Justfile"
mod jupyter ".taskfiles/jupyter/Justfile"


doc('List available recipes')
default:
    just --list


doc('Catch feature branch up to main')
catchup:
    "{{root_dir}}/scripts/git_catchup_main.sh" "{{root_dir}}"


doc('Git rebase on main')
_rebase:
    "{{root_dir}}/scripts/git_rebase_main.sh" "{{root_dir}}"
