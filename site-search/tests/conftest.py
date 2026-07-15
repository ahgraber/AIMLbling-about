from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
import pytest


@pytest.fixture(scope="session")
def potion_model_path() -> Path:
    """Resolve the model once, using cache first and fetching only when absent."""
    try:
        return Path(snapshot_download("minishlab/potion-base-8M", local_files_only=True))
    except LocalEntryNotFoundError:
        return Path(snapshot_download("minishlab/potion-base-8M"))
