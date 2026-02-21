# Repository Guidelines for AI Agents

This document provides essential context and instructions for AI agents working on this repository.

## Project Overview

This is a personal blog and experiments monorepo for organizing thoughts about LLMs, Generative AI, papers from arxiv.org, and other topics in data science and machine learning.

It contains two major areas:

- **Hugo blog**: A static site built with [Hugo](https://gohugo.io/) using the [Hextra](https://imfing.github.io/hextra/) theme, containerized and deployed to a homelab Kubernetes cluster.
- **Experiments**: Standalone Python experiments, each with its own environment managed by `uv`.

## Repository Structure

```text
.
├── AGENTS.md             # This file — repo guidelines for AI agents
├── flake.nix             # Nix flake providing Hugo dev environment
├── justfile              # Top-level just recipes (task runner)
├── pyproject.toml        # Root Python project config (uv workspace root)
├── docs/                 # Human-readable documentation (nix, containerization, etc.)
├── scripts/              # Utility shell scripts (git ops, hugo build/publish)
├── hugo/                 # Hugo static site
│   ├── justfile          # Hugo just module
│   ├── config/           # Hugo configuration
│   ├── content/          # Blog posts and pages (Markdown)
│   │   ├── blog/         # Blog posts (each in its own directory with index.md)
│   │   ├── treadmill/    # AI treadmill tracking data
│   │   └── about/        # About page
│   ├── layouts/          # Hugo templates and shortcodes
│   ├── assets/           # CSS and other assets
│   ├── docker/           # Dockerfile for containerized deployment
│   ├── public/           # Generated static site (build output)
│   └── themes/           # Hugo themes
└── experiments/          # Python experiments (each is a discrete project)
    ├── justfile          # Jupyter just module
    ├── aiml/             # Shared utilities package used by other experiments
    ├── bitter-lesson/    # Bitter lesson analysis
    ├── fractals/         # Fractal drawing
    ├── language-identification/  # Language model comparison
    ├── parameter-estimation/     # Parameter estimation
    ├── ragas-experiment/         # RAG evaluation with RAGAS
    ├── sk-rant/                  # Semantic Kernel experiment
    └── typos-experiment/         # Typo analysis and embeddings
```

## Nix Development Environment

This repository uses a Nix flake to provide a consistent development environment with Hugo and related tools (dart-sass, docker, podman, colima, qemu).

### Entering the Dev Shell

```bash
nix develop
```

If `nix develop` fails, add experimental feature flags:

```bash
nix --extra-experimental-features 'nix-command flakes' develop
```

Alternatively, use `direnv` with a `.envrc` containing `use flake` for automatic shell activation.

### Running Commands with Nix

To run a single command inside the dev shell without entering it interactively:

```bash
nix develop -c <command>
```

Examples:

```bash
# Start Hugo dev server with drafts
nix develop -c hugo server -D --disableFastRender -s hugo

# Build the site
nix develop -c hugo -s hugo
```

### Updating the Dev Shell

```bash
nix flake update
```

## Task Runner: just

This repo uses [`just`](https://github.com/casey/just) as its task runner.

### Using just

```bash
# List all available recipes
just --list

# List recipes in a module
just hugo --list
just jupyter --list
```

### Key Recipes

#### Hugo recipes (`just hugo <recipe>`)

| Recipe                         | Description                                                  |
| ------------------------------ | ------------------------------------------------------------ |
| `just hugo new title="..."`    | Create a new draft post on a dedicated branch                |
| `just hugo publish path="..."` | Set a draft to published (sets `draft: false`, updates date) |
| `just hugo demo`               | Serve the site locally with drafts enabled                   |
| `just hugo clean`              | Clean rendered static site content                           |
| `just hugo build`              | Build the container image (requires colima/docker)           |
| `just hugo run`                | Run the container locally                                    |
| `just hugo treadmill`          | Update treadmill Hugo module data                            |
| `just hugo catchup`            | Rebase feature branch onto main                              |

#### Jupyter recipes (`just jupyter <recipe>`)

| Recipe                           | Description                              |
| -------------------------------- | ---------------------------------------- |
| `just jupyter strip ipynb="..."` | Strip notebook metadata, keeping outputs |

#### Root recipes

| Recipe         | Description                     |
| -------------- | ------------------------------- |
| `just catchup` | Catch feature branch up to main |

### Important: Task Runner Usage for Agents

**Agents SHOULD use `just` recipes directly** — they wrap the underlying commands with correct paths and precondition checks.

If a recipe fails or you need to understand what it does, read the corresponding justfile (`hugo/justfile` or `experiments/justfile`) and then run the underlying commands manually inside the nix dev shell:

```bash
nix develop -c hugo server -D --disableFastRender -s hugo
```

## Python Environment

- **Package manager**: `uv` (defined in `pyproject.toml`).
- **Root project**: The root `pyproject.toml` defines the workspace; the base package is `experiments/aiml/`.
- **Experiments**: Each experiment directory under `experiments/` is a discrete project with its own `pyproject.toml` and dependencies.

### Activating the Default Environment

```bash
# Sync and activate from repo root (installs the default aiml package)
uv sync
```

### Working with a Specific Experiment

```bash
# Sync a specific experiment's environment
uv sync --project experiments/<dirname>
```

### Running Python

Always activate the `uv`-managed environment before running Python:

```bash
uv run python <script.py>
# or
uv run pytest
```

Do not install packages manually — if a required package is unavailable, alert the user.
