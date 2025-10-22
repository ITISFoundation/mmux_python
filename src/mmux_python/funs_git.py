import os
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import requests


def get_mmux_top_directory() -> Path:
    workspace_dir = Path(os.getcwd())
    if workspace_dir.name == "tests":
        workspace_dir = workspace_dir.parent
    assert workspace_dir.name == "mmux_gui"
    ## TODO this should eventually be renamed to MMUX
    return workspace_dir


def check_repo_exists(repo_url: str) -> bool:
    """
    Checks if the given GitHub repository URL exists.
    Returns True if the repository exists, otherwise False.
    """
    try:
        # Convert the Git URL to an API URL to check repository existence
        response = requests.head(repo_url)
        return response.status_code == 200
    except Exception:
        return False


def clone_repo(
    repo_url: str,
    target_dir: Path | None = None,
    commit_hash: str | None = None,
) -> tuple[bool, str]:
    if target_dir is None:
        target_dir = Path()

    # ## Optional: improve error cathcing w this library, or use sub-process below
    # repo_path = target_dir / repo_url.split("/")[-1]
    # git.Repo.clone_from(repo_url, repo_path)
    # repo = git.Repo(repo_path)

    result = subprocess.run(
        ["git", "clone", repo_url, str(target_dir)],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:  ## SUCCESS
        if commit_hash:
            checkout_result = subprocess.run(
                ["git", "-C", str(target_dir), "checkout", commit_hash],
                capture_output=True,
                text=True,
            )
            install_requirements(repo_path=target_dir)
            if checkout_result.returncode:
                return (True, f"Successfully checked out commit {commit_hash}")
            if checkout_result.returncode != 0:
                return (
                    False,
                    f"Error checking out commit {commit_hash}: {checkout_result.stderr.strip()}",
                )
        else:
            # Extract the commit hash from the result.stdout
            log_result = subprocess.run(
                ["git", "-C", str(target_dir), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
            )
            install_requirements(repo_path=target_dir)
            if log_result.returncode == 0:
                commit_hash = log_result.stdout.strip()
                return (
                    True,
                    f"Repository cloned successfully into {target_dir}. Current commit hash: {commit_hash}.",
                )
            else:
                return (
                    False,
                    f"Repository cloned successfully into {target_dir}, but failed to get commit hash: {log_result.stderr.strip()}",
                )

    else:  ## FAILURE
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        if "already exists and is not an empty directory" in result.stderr:
            return (
                False,
                f"Error: Target directory '{target_dir}/{repo_name}' already exists and is not empty.",
            )
        elif "could not resolve host" in result.stderr:
            return (
                False,
                "Error: Could not resolve host. Please check the repository URL.",
            )
        elif "Repository not found" in result.stderr:
            return (
                False,
                "Error: Repository not found. Please check the repository URL.",
            )
        else:
            return (
                False,
                f"Unknown error when cloning the repository: {result.stderr.strip()}",
            )
    return (False, "Unknown error occurred.")


def install_requirements(repo_path: Path) -> None:
    """Check if there is a requirements.* in the repo path, and pip install it if so."""
    requirements_files = list(repo_path.glob("requirements*.txt"))
    if not requirements_files:
        print("No requirements file found.")
        return

    for req_file in requirements_files:
        print(req_file)
        cmd = [sys.executable, "-m", "pip", "install", "-r", req_file.name]
        print(
            "Installation command: ",
            " ".join(cmd),
        )
        print("cwd = ", str(repo_path))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(repo_path),
        )
        print(result)
        print(result.stdout)
        print(result.stderr)
        print("returncode: ", result.returncode)
        assert (
            result.returncode == 0
        ), f"Failed to install requirements from {req_file}: {result.stderr.strip()}"
        print(result.stdout)

    print("Requirements installed successfully.")
    return


def get_attr_from_repo(repo_path: Path, module_name: str, function_name: str) -> Any:
    """
    Import a specific function from a module within the cloned repo.

    Args:
    - repo_path: Path to the cloned repository.
    - module_path: Relative path to the module (e.g., "evaluate.py").
    - function_name: Name of the function to import.

    Returns:
    - The imported element (variable, function, ...)
    """
    # Construct the full module path
    full_module_path = repo_path / module_name
    assert full_module_path.is_file(), f"Module {full_module_path!s} not found"

    # Load the module
    spec = spec_from_file_location("loaded_module", str(full_module_path))
    assert (
        spec is not None and spec.loader is not None
    ), f"module {module_name} not found in {full_module_path!s}"
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the element from the module
    func = getattr(module, function_name)
    return func
