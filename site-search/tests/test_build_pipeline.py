from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_local_recipe_builds_corpus_then_gates_and_exports_served_artifacts() -> None:
    justfile = (ROOT / "hugo" / "justfile").read_text(encoding="utf-8")

    recipe = justfile[justfile.index("search-index:") :]
    first_hugo = recipe.index('hugo -s "{{hugo_dir}}"')
    gate = recipe.index("python -m site_search.gate")
    export = recipe.index("python -m site_search.build_index")
    assert first_hugo < gate < export
    assert '--search-data "{{hugo_dir}}/public/en.search-data.json"' in recipe
    assert '--out "{{hugo_dir}}/static/search"' in recipe
    assert "Run `just hugo search-index` before `just hugo demo`" in justfile


def test_container_rebuilds_artifacts_from_its_first_hugo_pass_before_final_render() -> None:
    dockerfile = (ROOT / "hugo" / "docker" / "Dockerfile").read_text(encoding="utf-8")

    assert "COPY pyproject.toml uv.lock ./" in dockerfile
    assert "COPY experiments/aiml/pyproject.toml" not in dockerfile
    assert "COPY site-search/pyproject.toml ./site-search/pyproject.toml" in dockerfile
    assert "COPY site-search/src ./site-search/src" in dockerfile
    assert "uv sync --frozen --package site-search --no-default-groups" in dockerfile
    copy_hugo = dockerfile.index("COPY hugo ./hugo")
    clean = dockerfile.index("rm -rf ./hugo/public ./hugo/static/search")
    first_hugo = dockerfile.index("hugo -s ./hugo")
    gate = dockerfile.index("python -m site_search.gate")
    export = dockerfile.index("python -m site_search.build_index")
    final_hugo = dockerfile.index("hugo --minify -s ./hugo")
    # Anti-staleness invariant: the host hugo/ tree (which may carry a developer's
    # locally-built static/search) is copied first, then unconditionally cleaned
    # before the two-pass build regenerates artifacts from this build's content.
    # Reordering COPY after the clean would let stale host artifacts survive into
    # the image, so pin the order explicitly.
    assert copy_hugo < clean < first_hugo < gate < export < final_hugo
    assert "./hugo/public/en.search-data.json" in dockerfile
    assert "./hugo/static/search" in dockerfile


def test_search_pipeline_changes_trigger_container_publish() -> None:
    workflow = (ROOT / ".github" / "workflows" / "publish.yaml").read_text(encoding="utf-8")

    assert 'paths: ["hugo/**", "site-search/**", "pyproject.toml", "uv.lock"]' in workflow


def test_podman_build_uses_docker_format_and_positional_platform_override() -> None:
    script = (ROOT / "scripts" / "hugo_build_image.sh").read_text(encoding="utf-8")
    run_script = (ROOT / "scripts" / "hugo_run_local.sh").read_text(encoding="utf-8")
    justfile = (ROOT / "hugo" / "justfile").read_text(encoding="utf-8")

    assert "  --format docker\n" in script
    assert '--build-arg "BUILDARCH=${build_arch}"' in script
    assert 'image="ghcr.io/ahgraber/aimlbling-about:debug"' in script
    assert 'image="ghcr.io/ahgraber/aimlbling-about:debug"' in run_script
    assert 'podman manifest rm "${image}"' in script
    assert 'podman image rm "${image}"' in script
    assert 'build_cmd+=(--manifest "${image}")' in script
    assert "build_cmd+=(-t" not in script
    assert '[[ "${platforms}" == platforms=* ]]' in script
    assert "just hugo build linux/arm64" in script
    assert "just hugo build linux/arm64" in justfile
    assert 'platforms="linux/arm64"' not in script
    assert 'platforms="linux/arm64"' not in justfile


def test_darwin_podman_shim_health_checks_and_waits_for_machine_api() -> None:
    flake = (ROOT / "flake.nix").read_text(encoding="utf-8")

    assert "wait_for_podman()" in flake
    assert "${pkgs.coreutils}/bin/timeout 2" in flake
    assert 'deadline="$((SECONDS + 30))"' in flake
    assert "${podmanReal} info" in flake
    assert '[ "$SECONDS" -ge "$deadline" ]' in flake
    assert "did not become ready within 30 seconds" in flake
    assert "wait_for_podman || exit 1" in flake


def test_container_uses_build_arch_sass_and_managed_wheel_only_python() -> None:
    dockerfile = (ROOT / "hugo" / "docker" / "Dockerfile").read_text(encoding="utf-8")
    build_stage = dockerfile[: dockerfile.index("FROM docker.io/nginxinc")]

    assert "python3" not in dockerfile
    assert (
        "FROM --platform=$BUILDPLATFORM "
        "ghcr.io/astral-sh/uv:0.11.26@sha256:3d868e555f8f1dbc324afa005066cd11e1053fc4743b9808ca8025283e65efa5 AS uv"
        in dockerfile
    )
    assert "COPY --from=uv /uv /uvx /bin/" in dockerfile
    assert "ARG BUILDARCH" in dockerfile
    assert 'amd64) sass_arch="x64"' in dockerfile
    assert 'arm64) sass_arch="arm64"' in dockerfile
    assert "unsupported build architecture" in dockerfile
    assert "TARGETARCH" not in dockerfile
    assert "ENV UV_PYTHON=3.12.13" in dockerfile
    assert "ENV UV_MANAGED_PYTHON=1" in dockerfile
    assert 'RUN uv python install --no-bin "${UV_PYTHON}"' in dockerfile
    assert "ENV UV_PYTHON_DOWNLOADS=never" in dockerfile
    assert "ENV PYTHONPATH=/src/site-search/src" in dockerfile
    assert "--no-install-project" in dockerfile
    assert "--no-build" in dockerfile
    assert "musllinux wheel" in dockerfile
    assert "move the build stage to glibc" in dockerfile
    assert "LABEL org.opencontainers.image" not in build_stage

    dependency_sync = dockerfile.index("uv sync --frozen")
    assert dependency_sync < dockerfile.index("COPY .git ./.git")
    assert dependency_sync < dockerfile.index("COPY site-search/src ./site-search/src")
    assert dependency_sync < dockerfile.index("COPY hugo ./hugo")
