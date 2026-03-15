# Experiments

Each directory is a discrete experiment with its own `pyproject.toml`, managed as a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/) member.
A single `uv.lock` at the repo root resolves dependencies for all workspace members.

The shared utilities package `aiml/` provides common dependencies via optional extras (e.g., `aiml[ds,plot,nlp]`).

## Usage

```sh
# Scaffold a new experiment
just experiments new <name>

# List available experiments
just experiments list

# Sync an experiment's environment
just experiments sync <name>

# Run a command in an experiment's context
just experiments run <name> python <script.py>

# Run tests for an experiment
just experiments test <name>

# Upgrade all workspace dependencies and re-lock
just experiments upgrade
```

Or equivalently with `uv` directly:

```sh
uv sync --package <name>
uv run --package <name> python <script.py>
```

## Excluded member

`language-identification/` is excluded from the workspace due to conflicting constraints (`numpy<2`) and is managed independently:

```sh
cd experiments/language-identification
uv sync
```
