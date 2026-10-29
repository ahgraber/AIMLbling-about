"""Image build/push/run scripts drive the selected container CLI with the expected arguments.

Stub `docker` and `podman` executables record each invocation, so no container
runtime is needed. PATH holds only the stubs and the tools the scripts use, so a
real CLI on the host (CI runners ship docker) can never be selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / "scripts" / "hugo_build_image.sh"
RUN_SCRIPT = ROOT / "scripts" / "hugo_run_local.sh"
IMAGE = "ghcr.io/ahgraber/aimlbling-about:debug"
TOOLS = ("git", "date", "dirname", "mktemp", "awk", "rm", "cat")

# Records argv (one argument per line, then a separator) and whether TREADMILL_PAT
# reached the CLI's environment. Canned answers cover the podman queries. The run
# script deletes its temporary nginx config on exit, so STUB_CONF_COPY keeps a copy.
STUB = """#!/bin/sh
case "$*" in
  "info --format {{.Host.Arch}}") echo aarch64; exit 0 ;;
  "manifest exists"*) [ -n "${STUB_MANIFEST_EXISTS:-}" ]; exit $? ;;
  "image exists"*) [ -n "${STUB_IMAGE_EXISTS:-}" ]; exit $? ;;
esac
if [ -n "${STUB_CONF_COPY:-}" ]; then
  for a in "$@"; do
    case "$a" in *default.local.conf.*) cat "${a%%:*}" > "$STUB_CONF_COPY" ;; esac
  done
fi
{
  printf '%s\\n' "${0##*/}" "$@"
  printf 'PAT=%s\\n' "${TREADMILL_PAT:-<unset>}"
  printf -- '---\\n'
} >> "$STUB_LOG"
"""


@dataclass
class Call:
    argv: list[str]
    pat: str


@dataclass
class Result:
    returncode: int
    stderr: str
    calls: list[Call]


@dataclass
class Sandbox:
    """A throwaway repo root, the stub directory, and the tool directory PATH is limited to."""

    repo: Path
    tools: Path
    stubs: Path
    log: Path


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    repo = tmp_path / "repo"
    (repo / "hugo" / "docker").mkdir(parents=True)
    shutil.copy(ROOT / "hugo" / "docker" / "default.conf", repo / "hugo" / "docker" / "default.conf")
    git = shutil.which("git")
    assert git
    subprocess.run([git, "init", "-q", str(repo)], check=True)  # noqa: S603 -- absolute Git executable, fixed arguments.
    subprocess.run(  # noqa: S603 -- absolute Git executable, fixed arguments.
        [
            git,
            "-C",
            str(repo),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "init",
        ],
        check=True,
    )

    tools = tmp_path / "tools"
    tools.mkdir()
    for tool in TOOLS:
        found = shutil.which(tool)
        assert found, f"{tool} is required to run the image scripts"
        (tools / tool).symlink_to(found)

    stubs = tmp_path / "stubs"
    stubs.mkdir()
    return Sandbox(repo=repo, tools=tools, stubs=stubs, log=tmp_path / "calls.log")


def install_stubs(sandbox: Sandbox, *names: str) -> None:
    for name in names:
        stub = sandbox.stubs / name
        stub.write_text(STUB)
        stub.chmod(0o755)


def run(sandbox: Sandbox, script: Path, *args: str, **env: str) -> Result:
    bash = shutil.which("bash")
    assert bash
    environment = {
        "PATH": f"{sandbox.stubs}:{sandbox.tools}",
        "HOME": str(sandbox.repo),
        "STUB_LOG": str(sandbox.log),
        **env,
    }
    # The exit status is asserted by each test, so a failing script must not raise here.
    proc = subprocess.run(  # noqa: S603 -- absolute Bash executable running a repo script with test-chosen arguments.
        [bash, str(script), *args], check=False, env=environment, capture_output=True, text=True
    )
    calls = []
    if sandbox.log.exists():
        for block in sandbox.log.read_text().split("---\n"):
            lines = block.splitlines()
            if lines:
                calls.append(Call(argv=lines[:-1], pat=lines[-1].removeprefix("PAT=")))
    return Result(proc.returncode, proc.stderr, calls)


def build(sandbox: Sandbox, *args: str, **env: str) -> Result:
    repo = sandbox.repo
    return run(sandbox, BUILD_SCRIPT, str(repo), str(repo / "hugo"), *args, **env)


def option(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


# --- CLI selection ---


def test_selection__docker_and_podman_on_path__uses_docker(sandbox: Sandbox) -> None:
    install_stubs(sandbox, "docker", "podman")

    result = build(sandbox, "build")

    assert result.returncode == 0, result.stderr
    assert [call.argv[0] for call in result.calls] == ["docker"]


def test_selection__only_podman_on_path__uses_podman(sandbox: Sandbox) -> None:
    install_stubs(sandbox, "podman")

    result = build(sandbox, "build")

    assert result.returncode == 0, result.stderr
    assert {call.argv[0] for call in result.calls} == {"podman"}


def test_selection__aimlbling_engine_set__overrides_preference(sandbox: Sandbox) -> None:
    install_stubs(sandbox, "docker", "podman")

    result = build(sandbox, "build", AIMLBLING_ENGINE="podman")

    assert result.returncode == 0, result.stderr
    assert {call.argv[0] for call in result.calls} == {"podman"}


def test_selection__unknown_engine__fails_without_calling_a_cli(sandbox: Sandbox) -> None:
    install_stubs(sandbox, "docker")

    result = build(sandbox, "build", AIMLBLING_ENGINE="container")

    assert result.returncode == 2
    assert "Unknown AIMLBLING_ENGINE 'container'" in result.stderr
    assert result.calls == []


# --- docker backend ---


def test_docker_build__no_platform__loads_native_image_without_platform_flag(sandbox: Sandbox) -> None:
    install_stubs(sandbox, "docker")

    result = build(sandbox, "build")

    assert result.returncode == 0, result.stderr
    (call,) = result.calls
    assert call.argv[1:3] == ["buildx", "build"]
    assert "--load" in call.argv
    assert "--push" not in call.argv
    assert "--platform" not in call.argv
    assert option(call.argv, "--tag") == IMAGE
    assert option(call.argv, "-f") == str(sandbox.repo / "hugo" / "docker" / "Dockerfile")
    assert call.argv[-1] == str(sandbox.repo)


def test_docker_build__platform_given__loads_that_platform(sandbox: Sandbox) -> None:
    install_stubs(sandbox, "docker")

    result = build(sandbox, "build", "linux/amd64")

    assert result.returncode == 0, result.stderr
    (call,) = result.calls
    assert option(call.argv, "--platform") == "linux/amd64"
    assert "--load" in call.argv


def test_docker_push__default_platforms__pushes_both_arches_without_loading(sandbox: Sandbox) -> None:
    install_stubs(sandbox, "docker")

    result = build(sandbox, "push")

    assert result.returncode == 0, result.stderr
    (call,) = result.calls
    assert call.argv[1:3] == ["buildx", "build"]
    assert option(call.argv, "--platform") == "linux/amd64,linux/arm64"
    assert "--push" in call.argv
    assert "--load" not in call.argv
    assert option(call.argv, "--tag") == IMAGE


# --- build secret ---


def test_secret__pat_unset__no_secret_flag(sandbox: Sandbox) -> None:
    install_stubs(sandbox, "docker")

    result = build(sandbox, "build")

    (call,) = result.calls
    assert "--secret" not in call.argv


def test_secret__pat_in_environment__passed_by_env_reference(sandbox: Sandbox) -> None:
    install_stubs(sandbox, "docker")

    result = build(sandbox, "build", TREADMILL_PAT="from-env")

    (call,) = result.calls
    assert option(call.argv, "--secret") == "id=treadmill_pat,env=TREADMILL_PAT"
    assert call.pat == "from-env"
    assert not any("from-env" in arg for arg in call.argv)


def test_secret__pat_in_dotenv__exported_to_the_cli(sandbox: Sandbox) -> None:
    install_stubs(sandbox, "docker")
    (sandbox.repo / ".env").write_text("TREADMILL_PAT=from-dotenv\n")

    result = build(sandbox, "build")

    (call,) = result.calls
    assert option(call.argv, "--secret") == "id=treadmill_pat,env=TREADMILL_PAT"
    assert call.pat == "from-dotenv"


# --- podman backend ---


def test_podman_build__native__tags_image_with_explicit_buildarch(sandbox: Sandbox) -> None:
    install_stubs(sandbox, "podman")

    result = build(sandbox, "build")

    assert result.returncode == 0, result.stderr
    (call,) = result.calls
    assert call.argv[:4] == ["podman", "build", "--format", "docker"]
    assert "BUILDARCH=arm64" in call.argv
    assert option(call.argv, "--tag") == IMAGE
    assert "--manifest" not in call.argv


def test_podman_build__manifest_left_by_push__removes_it_first(sandbox: Sandbox) -> None:
    install_stubs(sandbox, "podman")

    result = build(sandbox, "build", STUB_MANIFEST_EXISTS="1")

    assert result.returncode == 0, result.stderr
    assert [call.argv[1:3] for call in result.calls] == [["manifest", "rm"], ["build", "--format"]]


def test_podman_push__image_left_by_build__removes_it_before_the_manifest(sandbox: Sandbox) -> None:
    install_stubs(sandbox, "podman")

    result = build(sandbox, "push", STUB_IMAGE_EXISTS="1")

    assert result.returncode == 0, result.stderr
    assert [call.argv[1:3] for call in result.calls] == [["image", "rm"], ["build", "--format"], ["manifest", "push"]]
    assert result.calls[0].argv[3] == IMAGE


def test_podman_push__default_platforms__builds_manifest_then_pushes_all(sandbox: Sandbox) -> None:
    install_stubs(sandbox, "podman")

    result = build(sandbox, "push")

    assert result.returncode == 0, result.stderr
    build_call, push_call = result.calls
    assert option(build_call.argv, "--platform") == "linux/amd64,linux/arm64"
    assert option(build_call.argv, "--manifest") == IMAGE
    assert "BUILDARCH=arm64" in build_call.argv
    assert push_call.argv == ["podman", "manifest", "push", "--all", IMAGE, f"docker://{IMAGE}"]


# --- platform validation ---


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("build", "linux/amd64,linux/arm64"), "build loads one platform"),
        (("push", "linux/riscv64"), "Unsupported platform 'linux/riscv64'"),
        (("push", "linux/amd64,"), "Empty entry in platform list"),
        (("push", "platforms=linux/arm64"), "Platform override is positional"),
        (("deploy",), "Unknown mode 'deploy'"),
    ],
)
def test_validation__bad_mode_or_platforms__fails_before_calling_a_cli(
    sandbox: Sandbox, args: tuple[str, ...], message: str
) -> None:
    install_stubs(sandbox, "docker")

    result = build(sandbox, *args)

    assert result.returncode == 2
    assert message in result.stderr
    assert result.calls == []


# --- run ---


def test_run__default_port__maps_8081_to_container_8080_with_local_conf(sandbox: Sandbox) -> None:
    install_stubs(sandbox, "docker")
    conf_capture = sandbox.repo / "captured.conf"

    result = run(sandbox, RUN_SCRIPT, str(sandbox.repo / "hugo"), STUB_CONF_COPY=str(conf_capture))

    assert result.returncode == 0, result.stderr
    (call,) = result.calls
    assert call.argv[:5] == ["docker", "run", "--rm", "-p", "8081:8080"]
    mount = option(call.argv, "-v")
    assert mount is not None
    assert mount.endswith(":/etc/nginx/conf.d/default.conf:ro")
    assert call.argv[-1] == IMAGE
    conf = conf_capture.read_text()
    assert "listen 8080;" in conf
    assert "listen 80;" not in conf
    assert "return 301 https://" not in conf
    assert "absolute_redirect off;" in conf
    assert list((sandbox.repo / "hugo" / "docker").glob("default.local.conf.*")) == []
