from pathlib import Path
from utils.funs_evaluate import create_run_dir
from utils.funs_git import (
    clone_repo,
    get_attr_from_repo,
    check_repo_exists,
    get_mmux_top_directory,
)
from dotenv import dotenv_values
from typing import Callable, Tuple, Optional
import pandas as pd


def clone_spinal_repo(output_dir: Optional[Path] = None) -> Tuple[Path, str]:
    workspace_dir = get_mmux_top_directory()
    if output_dir is None:
        output_dir = create_run_dir(workspace_dir, "evaluation")
    config = dotenv_values("runner_spinal.env")
    repo_url = config["RUNNER_CODE_REPO"]
    assert repo_url is not None, f"Repository {repo_url} must be provided"
    assert check_repo_exists(repo_url)
    clone_repo(
        repo_url=repo_url,
        commit_hash=config["RUNNER_CODE_HASH"],
        target_dir=output_dir,
    )
    return output_dir, repo_url


def get_model_from_spinal_repo(output_dir: Optional[Path] = None) -> Callable:
    tmp_dir, repo_url = clone_spinal_repo(output_dir)
    fun: Callable = get_attr_from_repo(
        tmp_dir, module_name="evaluation.py", function_name="model"
    )
    return fun


## e.g. previously called "load_data_dakota_free_pulse_optimization"
def postpro_spinal_samples(df: pd.DataFrame) -> pd.DataFrame:
    ## Add objective functions with proper names; leave the old ones as columns as well
    if "1-activation" in df.columns:
        df["Activation (%)"] = 100 * (1.0 - df["1-activation"])
        df.pop("1-activation")
    elif "activation" in df.columns:
        df["Activation (%)"] = 100 * df["activation"]
        df.pop("activation")

    if "1-fsi" in df.columns:
        df["FSI"] = 1 - df["1-fsi"]
        df.pop("1-fsi")
    elif "fsi" in df.columns:
        df["FSI"] = df["fsi"]
        df.pop("fsi")

    if "energy" in df.columns:
        df["Energy"] = df["energy"]
        df.pop("energy")

    if "maxampout" in df.columns:
        df["Maximum Amplitude"] = df["maxampout"]
        df.pop("maxampout")
    elif "maxamp" in df.columns:
        df["Maximum Amplitude"] = df["maxamp"]
        df.pop("maxamp")

    return df
