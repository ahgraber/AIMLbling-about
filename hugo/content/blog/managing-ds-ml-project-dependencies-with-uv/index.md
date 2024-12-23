---
title: Managing DS/ML Project Dependencies With UV
date: 2024-12-14T16:01:57-05:00
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags:
  # meta
  - "blogumentation"
series: []
layout: single
toc: true
math: false
draft: true
---

Refer to [uv cli reference](https://docs.astral.sh/uv/reference/cli/)

## Init

Command: `uv init [OPTIONS] [PATH]`

`uv` supports 2 project archetypes:

- **app** - `uv init --app` - (default) web servers, scripts, and command-line interfaces
  - not intended to be built as python package, but can be built with `--package` option
- **lib** -`uv init --lib` - a project that provides functions and objects for other projects to consume
  - intended to be built and distributed as a Python package; implies `--package`

> Don't forget to activate the environment!
>
> ```sh
> source .venv/bin/activate # macOS, Linux
> .venv\Scripts\activate # windows
> ```

## Environment management

`uv` creates and manages virtual environments in a `.venv` directory next to a paired `pyproject.toml`

Project/package dependencies are added directly

```sh
uv add <packagename>
uv remove <packagename>
```

Development dependencies are local-only and are not runtime requirements.
They can be added using `--dev` flag.
Dependency groups are ways to organize development dependencies

```sh
uv add --dev pytest
uv add --group test pytest # adds in [test] dev dependency group
```

By default, `uv` includes only the `dev` dependency group in the environment (e.g., during uv run or uv sync).
The default groups to include can be changed using the `tool.uv.default-groups` setting.

```toml title="pyproject.toml"
[tool.uv]
default-groups = ["dev", "foo"]
```

To add an editable dependency (e.g., to install local pseudopackage into environment)

```sh
uv add --editable ./path/foo
```

### Locking and Syncing

The environment can be updated after updating the pyproject.toml dependencies with

```sh
uv sync
```

This will also update the `uv.lock` file.

#### Upgrading

To upgrade all packages:

```sh
uv lock --upgrade
```

To upgrade a single package to a specific version:

```sh
uv lock --upgrade-package <package>==<version>
```

### Jupyter

See [uv documentation on Jupyter integration](https://docs.astral.sh/uv/guides/integration/jupyter)

```sh
uv add --dev ipykernel ipywidgets notebook
```

To use with jupyter

```sh
# register kernel for project
uv run ipython kernel install --user --name=<project>
# start the jupyter server
uv run --with jupyter jupyter lab
```

## Monorepos / Multiproject repos

### Workspaces

Workspaces organize large codebases by splitting them into multiple packages with common dependencies.
Workspaces are intended to facilitate the development of multiple interconnected packages within a single repository;
In a workspace, each package defines its own pyproject.toml, but the workspace shares a single lockfile, ensuring that the workspace operates with a consistent set of dependencies.

To add a workspace to the project, update the `pyproject.toml` with table `[tool.uv.workspace]`

```toml title="pyproject.toml"
[project]
name = "uv-demo"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["pandas"]

[tool.uv.sources]
uv-demo = { workspace = true }

[tool.uv.workspace]
members = ["notebooks/*"]
exclude = []
```

> Every directory included by the members globs (and not excluded by the exclude globs) must contain a pyproject.toml file.

By default, `uv run` and `uv sync` operates on the workspace _root_.
To run in a specific workspace, use `uv run --package <name>`.

Dependencies specified in `[tool.uv.sources]` of the workspace root apply to all members (unless overridden on a per-member basis)

### Path dependencies

Workspaces are _not_ suited for cases in which members have conflicting requirements, or desire a separate virtual environment for
each member.
In this case, path dependencies may be a better option.

```toml title="pyproject.toml" hl_lines="8"
[project]
name = "experiment"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["pandas"]

[tool.uv.sources]
experiment = { path = "src/uv-demo" } # this applies to all members
```

An editable installation is not used for path dependencies by default. An editable installation may be requested for project directories:

```sh
uv add --editable ~/projects/bar/
```
