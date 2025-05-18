# %% Imports
import datetime
import logging
import os
from pathlib import Path
import platform
import subprocess

# %%
logger = logging.getLogger(__name__)
LOG_FMT = "%(asctime)s - %(levelname)-8s - %(name)s - %(funcName)s:%(lineno)d - %(message)s"


def basic_log_config() -> None:
    """Configure logging defaults."""
    logging.basicConfig(format=LOG_FMT)


# %%
# runtime utils
def this_file() -> str:
    """Get __file__ equivalent."""
    # try default, ipynb in vscode, ipynb in jupyter
    file_vars = ["__file__", "__vsc_ipynb_file__", "__session__"]
    for f in file_vars:
        if f in globals():
            file = Path(globals()[f]).expanduser()
            if file.exists():
                logger.debug(f"Found {f} at {file}")
                return str(file)  # return in string format like __file__

    raise RuntimeError("Unable to determine current file path")


def is_notebook() -> bool:
    """Check if running in notebook environment."""
    from IPython.core.getipython import get_ipython

    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def is_interactive():
    """Check if running in interactive repl."""
    import __main__ as main

    return not hasattr(main, "__file__")


def is_windows():
    """Identify whether system requires kerberos authentication."""
    return platform.system() == "Windows"


def torch_device():
    """Identify which device pytorch will use."""
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps:0")
    else:
        device = torch.device("cpu")

    logging.info(f"Found pytorch device '{device.type}'")
    return device


# %%
# path utils
def pathify(s: Path | str | None) -> Path:
    """Get path from string."""
    if s is None:
        raise ValueError("'s' cannot be None.")

    try:
        # preferred but undocumented in python < 3.11
        return Path(s).expanduser().absolute()
    except AttributeError:
        # resolve relative to cwd()
        return Path(s).resolve()


def is_drive(drive: Path | str) -> bool:
    """Return `True` if path is the drive root.

    Parameters
    ----------
    drive : path-like
    """
    p = Path(drive)
    return str(p) == str(p.parent) == p.anchor  # == p.drive + p.root


def get_repo_path(location: Path | str) -> Path:
    """Detect root of repo relative to given location by identifying .git/ directory.

    Parameters
    ----------
    location : pathlike
        `Path.cwd()` from the calling script

    Examples
    --------
    >>> if (__name__ == '__main__') and (__package__ is None):
    >>>     REPO_ROOT = get_repo_path(Path.cwd())
    >>>     sys.path.insert(0, str(REPO_ROOT / 'src'))
    # >>>     sys.path.append(str(REPO_ROOT / 'src'))
    """
    p = pathify(location)
    # we need a directory -- either the input dir or parent of input file
    cwd = p if p.is_dir() else p.parent

    # use git's cli to get the repo
    tool = get_tool("git")
    cmd = [tool, "rev-parse", "--show-toplevel"]
    result = subprocess.run(cmd, cwd=cwd, capture_output=True)  # NOQA: S603

    try:
        result.check_returncode()  # raises error if failed
    except subprocess.CalledProcessError as e:
        raise FileNotFoundError(f"Git repository not detected in {location} directory tree.") from e

    return pathify(result.stdout.decode("utf-8").strip())


def get_tool(util: Path | str) -> Path:
    """Get cli utility path."""
    from shutil import which

    tool = which(util, mode=os.X_OK, path=os.environ.get("PATH"))
    return pathify(tool)


# %%
