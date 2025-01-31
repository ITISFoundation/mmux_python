from pathlib import Path
from utils.funs_evaluate import create_run_dir
from utils.funs_git import (
    clone_repo,
    get_attr_from_repo,
    check_repo_exists,
    get_mmux_top_directory,
)
from dotenv import dotenv_values
from typing import Callable, Tuple


def clone_optistim_repo() -> Tuple[Path, str]:
    workspace_dir = get_mmux_top_directory()
    config = dotenv_values(workspace_dir / "runner_optistim.env")
    temp_dir = create_run_dir(workspace_dir, "evaluation")
    repo_url = config["RUNNER_CODE_REPO"]
    assert repo_url is not None, f"Repository {repo_url} must be provided"
    assert check_repo_exists(repo_url)
    clone_repo(
        repo_url=repo_url,
        commit_hash=config["RUNNER_CODE_HASH"],
        target_dir=temp_dir,
    )
    return temp_dir, repo_url


def get_model_from_optistim_repo() -> Callable:
    tmp_dir, repo_url = clone_optistim_repo()
    fun: Callable = get_attr_from_repo(
        tmp_dir, module_name="evaluation.py", function_name="model"
    )
    return fun


def test_clone_repo():
    ## check no issue when running the fixture
    tmp_dir, repo_url = clone_optistim_repo()
    assert repo_url == "https://github.com/ITISFoundation/optistim-pulse-evaluation"


def test_get_fun_from_repo():
    model: Callable = get_model_from_optistim_repo()
    assert list(model.__annotations__["inputs"].keys())[0] == "p1"
    assert list(model.__annotations__["outputs"].keys())[1] == "energy"
