from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from site_search.export import ARTIFACT_NAMES


def test_every_generated_search_artifact_is_gitignored() -> None:
    root = Path(__file__).resolve().parents[2]
    artifact_paths = [f"hugo/assets/search/{name}" for name in ARTIFACT_NAMES]
    git = shutil.which("git")
    assert git is not None

    completed = subprocess.run(  # noqa: S603 -- absolute Git executable with fixed arguments and artifact constants.
        [git, "check-ignore", "--no-index", *artifact_paths],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == artifact_paths
