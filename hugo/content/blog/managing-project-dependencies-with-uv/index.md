---
title: Managing Project Dependencies With `uv`
date: 2025-01-12
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags:
  # meta
  - blogumentation
  - opinion
series: []
layout: single
toc: true
math: false
draft: false
---

I've recently started using [uv](https://github.com/astral-sh/uv) to manage my python dependencies.
Previously, I primarily managed my dependencies with `conda`, and supplemented with `pip` when packages were not available through anaconda.org.
This process update is the result of workflow friction when using `conda` - it was unpleasant to jump through the required hoops myself, and unreasonable to expect others to do it when had such issues.

## Bill of Grievances

1. Not all packages are available through `conda`

2. `environment.yaml` allows installing with pip [_but does not support pip flags_](https://github.com/conda/conda/issues/6805)

3. `conda` environments (and surrounding ecosystem) do not lend themselves to "workspace"-style repos

   - `conda` creates globally-available environments
   - `direnv` and `vscode` assume a single environment per project and automatically activate it

4. `conda` ecosystem confusion - `conda`, `mamba`, `miniconda`, `micromamba` each attempt to provide the same functionality but do not reach 100% compatibility.

   > "You can swap _almost all_ commands between conda & mamba" (emphasis mine)\
   > "`micromamba` supports a subset of all `mamba` or `conda` commands"\
   > "While `micromamba` supports `conda-lock` "unified" lock files, Mamba currently does not."
   >
   > - [Mamba User Guide](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html)

The result of these issues is that it is hard to maintain a standard environment because you have to jump through hoops:
First, install or update your conda environment. Then `pip install -r requirements.txt` your dependencies that aren't distributed through anaconda.
Do your pip dependencies have requirements?
They might override or replace your conda-installed dependencies unless you specify `--no-deps`,
but then you have to ensure the dependencies of the pip-installed packages are already installed through conda or additionally specified in the `requirements.txt` file.
Forget `--no-deps`? Start over. 🤦🏻‍♂️

## Unified dependency management with `uv`

![uv](./images/uv.png)

`uv` makes the workflow much simpler. Dependencies are defined in `pyproject.toml` or can reuse an existing `requirements.txt`.
Virtual environments are automatically created and used. `uv` can manage and install python versions independent of system python.
Workspaces allow for dependency inheritance and caches for fast switching and to avoid re-downloading and re-building packages.
`uv` implements dependency locking for reproducibility.

## `uv` cliffnotes

Refer to [uv cli reference](https://docs.astral.sh/uv/reference/cli/)

### Init

Command: `uv init [OPTIONS] [PATH]`

`uv` supports 2 project archetypes:

- **app** - `uv init --app` - (default) web servers, scripts, and command-line interfaces
  - not intended to be built as python package, but can be built with `--package` option
- **lib** -`uv init --lib` - a project that provides functions and objects for other projects to consume
  - intended to be built and distributed as a Python package; implies `--package`

> [!TIP]
> Don't forget to activate the environment!
>
> ```sh
> source .venv/bin/activate # macOS, Linux
> .venv\Scripts\activate    # windows
> ```

### [Environment management](https://docs.astral.sh/uv/concepts/projects/dependencies/#adding-dependencies)

`uv` creates and manages virtual environments in a `.venv` directory next to a paired `pyproject.toml` (i.e., in the same folder as your project, as opposed to `conda`'s global/project-independent location)

Project/package dependencies are added directly

```sh
uv add package1 'package2>=0.4.2'
uv remove package
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

Optional dependencies can be added similarly:

```sh
uv add --optional group package # add 'package' to 'group' optional dependency group
```

To add an editable dependency (e.g., to install local pseudopackage into environment)

```sh
uv add --editable ./path/foo
```

#### Locking and Syncing

The environment can be updated after updating the pyproject.toml dependencies with

```sh
uv sync
```

This will also update the `uv.lock` file.

##### Upgrading

To upgrade all packages in the environment:

```sh
uv lock --upgrade
```

To upgrade a single package to a specific version:

```sh
uv lock --upgrade-package <package>==<version>
```

#### Jupyter

See [uv documentation on Jupyter integration](https://docs.astral.sh/uv/guides/integration/jupyter)

```sh
uv add --dev ipykernel ipywidgets notebook
```

To use with jupyter (this is unnecessary if using notebooks or `#%%` scripts in vscode)

```sh
# register kernel for project
uv run ipython kernel install --user --name=<project>
# start the jupyter server
uv run --with jupyter jupyter lab
```

### Monorepos / Multiproject repos

#### [Workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/)

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

By default, `uv run` and `uv sync` operates on the _workspace root_ (i.e., the `pyproject.tomml` defined at the project root).
To use the environment of a specific workspace, use the name defined in `[tool.uv.sources]`:

```sh
# update the dependencies for the workspace
uv sync --package <workspace_name>
# run a command in a workspace with workspace dependencies
uv run --package <workspace_name> <command>
```

Dependencies specified in `[tool.uv.sources]` of the workspace root apply to all members (unless overridden on a per-member basis)

> [!NOTE]
> Workspace switching takes advantage of the fact that direnv/vscode/etc. assume a single environment per repo.
> Updating the workspace updates the single repo `.venv` directory (this is fast because `uv` caches dependencies).
> This means that vscode, for instance, will always use the expected workspace dependencies when working in a repo because it uses the single `.venv` location;
> it is only the contents of the virtual environment that change when switching workspaces.

#### Path dependencies

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
uv add --editable /path/to/projects/bar/
```

## Example - Workspaces with central dependency

As an example, I'll share the configuration I use for this blog:

1. I have a local package `aiml` where I define convenience functions used across all of my experiments.
   Due to the organizational decision to separate blog content (`hugo/`) from experiments (`experiments/`), I don't use a standard `src` style layout.
   Typically, the main local package would be at `src/<packagename>/`.
2. Each experiment has a different set of dependencies and is managed as a workspace.
3. When possible, workspaces use the same dependencies/versions.

As a result, I end up with a fairly complex configuration:

- The `pyproject.toml` at the repo root defines a "default" python environment based on the local `aiml` package (1, 3).
- `experiments\aiml\pyproject.toml` defines the local `aiml` package (1) and sets of optional dependencies for easy installation (3)
- `experiments\*\pyproject.toml` define the workspace dependencies, and typically leverate the predefined sets with `aiml[set1, set2,...]` (2, 3).
  If needed, the workspace can define additional dependencies or override the optional dependencies specified in `aiml`.

{{< tabs items="root,main,workspace" >}}

{{< tab >}}

This is the `pyproject.toml` at the root of the repo:

```toml
# --- project ----------------------------------------------------------------
[project]
name = "aimlbling-about"
version = "v0.beta"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.10,<3.13"
dependencies = [
  "aiml[ds,db,plot]", # default venv
  "pydantic>=2",
]

[dependency-groups]
dev = ["aiml[dev]"]
lint = ["aiml[lint]"]
test = ["aiml[test]"]

# --- uv ---------------------------------------------------------------------
# This file serves as the root pyproject.toml definition.
# The base project for the workspace is `./experiments/aiml`

[tool.uv]
package = false
default-groups = ["dev", "test"]
python-preference = "managed"

[tool.uv.sources]
aiml = { workspace = true }

torch = [
  { index = "pytorch-cpu", marker = "platform_system != 'Darwin'" },
  # { index = "pytorch-gpu", extra="gpu", marker = "platform_system != 'Darwin'" },
]

[tool.uv.workspace]
members = ["./experiments/*"]
exclude = ["./docs", "./hugo"]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-gpu"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

{{< /tab >}}
{{< tab >}}

This is the `experiments\aiml\pyproject.toml` for the "main" codebase; we assume this local package will be available in all other workspaces
and declare a standardized set of dependencies for the project that can be used per-workspace

```toml
[project]
name = "aiml"
version = "0.0.1"
description = "Utilities for AI/ML experiments"
requires-python = ">=3.10, <3.13"
dependencies = [
  "ipython>=8.8",
  "ipywidgets>=8.1.5",
  "notebook>=7.3.1",
  "pydantic>=2",
  "python-dateutil>=2.9.0.post0",
  "python-dotenv>=1.0.1",
  "typing-extensions>=4.12",
]

[dependency-groups]
dev = ["aiml[dev]"]
lint = ["aiml[lint]"]
test = ["aiml[test]"]

[project.optional-dependencies]
dev = ["hatchling>=1.26.3"]
test = [
  "coverage>=4.2",
  "pytest>=8.3.4",
  "pytest-asyncio>=0.24.0",
  "pytest-cov>=6.0.0",
]
ds = [
  # ref: https://scientific-python.org/specs/spec-0000/
  "jinja2>=3.1",
  "joblib>=1.4",
  "networkx>3.1",
  "numpy>=1.26",
  "openpyxl>=3.1",
  "pandas>=2",
  "python-dateutil>=2.9",
  "python-dotenv>=1.0",
  "pyyaml>=6.0.2",
  "scikit-learn>=1.3",
  "scipy>=1.12",
  "tabulate>=0.9.0",
  "xlrd>=2.0",
]
db = ["pyodbc>=5", "sqlalchemy>=2"]
plot = [
  "cmcrameri>=1.9",
  "matplotlib>=3.8",
  "mizani>=0.13",
  "plotly>=5.24",
  "plotnine>=0.14",
  "seaborn>=0.13",
]
api = ["fastapi>=0.115.6", "uvicorn>=0.32.1"]
nlp = [
  "langcodes>=3.5",
  "lingua-language-detector>=2.0",
  "nltk>=3.9",
  "rouge-score>=0.1.2",
  "spacy>=3.7,<4",
]
torch = [
  "tensorboard>=2.18",
  "torch>=2",
  "torchaudio>=2.5",
  "torchdata>=0.10",
  "torchmetrics>=1.6",
  "torchtext>=0.18",
  "torchvision>=0.20",
]
transformers = [
  "accelerate>=1.2.0",
  "datasets>=3.2.0",
  "huggingface-hub>=0.23",
  "safetensors>=0.4.5",
  "sentence-transformers>=3.3.1",
  "sentencepiece>=0.2.0",
  "text-generation>=0.7.0",
  "tokenizers>=0.20,<1",
  "transformers>=4.46,<5",
]
langchain = [
  "langchain>=0.3",
  "langchain-anthropic>=0.3.0",
  "langchain-community>=0.3",
  "langchain-experimental>=0.3.3",
  "langchain-huggingface>=0.1.2",
  "langchain-openai>=0.2",
  "langchain-text-splitters>=0.3",
  "langchain-together>=0.2.0",
  "langchain-voyageai>=0.1.3",
]
llm = [
  # "aisuite[all]>=0.1.3",
  "bert-score>=0.3.13",
  "bm25s>=0.2.5",
  "dspy-ai>=2.5",
  "jiter>=0.8",
  "json-repair>=0.30",
  "openai>=1.57",
  "openapi-core>=0.19",
  "openapi-schema-validator>=0.6",
  "outlines>=0.1.10",
  "pydantic-ai>=0.0.2",
  "ragas>=0.2",
  "semantic-kernel>=1.17",
  "tenacity>=8.5",
  "tiktoken>=0.8",
  "together>=1.3.5",
  "unstructured>=0.16.11",
  "voyageai>=0.3.2",
]
llamaindex = [
  "llama-index-agent-openai>=0.4.0",
  "llama-index>=0.11,<0.13",
  "llama-index-embeddings-huggingface>=0.4.0",
  "llama-index-embeddings-openai>=0.3.1",
  "llama-index-embeddings-together>=0.3.0",
  "llama-index-embeddings-voyageai>=0.3.1",
  "llama-index-llms-anthropic>=0.5.0",
  "llama-index-llms-huggingface>=0.4.1",
  "llama-index-llms-huggingface-api>=0.3.0",
  "llama-index-llms-openai-like>=0.3.3",
  "llama-index-llms-together>=0.3.1",
  "llama-index-program-openai>=0.3.1",
  "llama-index-question-gen-openai>=0.3.0",
  "llama-index-readers-file>=0.4.1",
  "llama-index-readers-llama-parse>=0.4.0",
  "llama-index-retrievers-bm25>=0.5.0",
  "llama-index-vector-stores-duckdb>=0.3.0",
]

[tool.uv]
package = true
default-groups = ["dev", "test"]
python-preference = "managed"

# --- build-system -----------------------------------------------------------
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

# ref: https://hatch.pypa.io/1.2/version/#configuration
[tool.hatch.version]
path = "./src/VERSION"
pattern = "^(?P<version>.+?)(\n)"

[tool.hatch.build]
only-include = ["./src/VERSION", "./src/aiml", "tests"]
skip-excluded-dirs = true
# sources = ["src"]

[tool.hatch.build.targets.sdist]

[tool.hatch.build.targets.wheel]
packages = ["src/aiml"]
macos-max-compat = true
```

{{< /tab >}}
{{< tab >}}

This is the `experiments\*\pyproject.toml` of a workspace.

```toml
[project]
name = "ragas-experiment"
version = "0.0.1"
description = ""
requires-python = ">=3.10, <3.13"
dependencies = [
  "aiml[ds,db,plot,nlp,torch,transformers,langchain,llm,llamaindex]",
  # additional required deps here
]

[dependency-groups]
dev = ["aiml[dev]"]
lint = ["aiml[lint]"]
test = ["aiml[test]"]
```

{{< /tab >}}
{{< /tabs >}}
