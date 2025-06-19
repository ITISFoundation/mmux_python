from typing import List, Optional, Callable, Dict, TypeVar, overload
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import copy
import re

def _parse_data(file: str | Path) -> List[List[str]]:
    """
    Parse a space-delimited text file into a list of lists.
    
    Args:
        file: Path to the file to parse
        
    Returns:
        List of lines, where each line is split into a list of values
    """
    data = []
    with open(file) as f:
        data = [line.strip().split() for line in f]
    return data


def _parse_json_dict(file: str | Path):
    print("DEPRECATED! _parse_json_dict was used with the old ParallelRunner input files")
    with open(file) as f:
        data_dict = json.load(f)["tasks"]

    columns = list(data_dict[0]["input"]["InputFile1"]["value"].keys())
    ## FIXME manual fix, we should be using all output keys
    # columns += list(data_dict[0]["output"]["OutputFile1"]["value"].keys())
    columns += ["-AFpeak"]

    data = []
    for d in data_dict:
        input_data = list(d["input"]["InputFile1"]["value"].values())
        ## FIXME manual fix, we should be using all output keys
        # output_data = list(d["output"]["OutputFile1"]["value"].values())
        output_data = [d["output"]["OutputFile1"]["value"]["AFmax_4um"]]

        data.append(input_data + output_data)

    return columns, data


def process_json_file(file: str) -> str:
    columns, data = _parse_json_dict(file)
    df = pd.DataFrame(data, columns=columns)
    df[r"%eval_id"] = np.arange(1, len(df) + 1)
    df = df[
        [r"%eval_id"] + [col for col in df.columns if col != r"%eval_id"]
    ]  # move eval_id to the front

    processed_file = file.replace(".json", "_json.txt")
    df.to_csv(processed_file, sep=" ", index=False)
    return processed_file


def get_variable_names(file: str | Path) -> List[str]:
    file = Path(file)
    assert file.exists(), f"File {file} does not exist"
    _, ext = os.path.splitext(file)
    if ext == ".dat" or ext == ".txt":
        lines = _parse_data(file)
        return sanitize_varnames(lines[0])
    elif ext == ".json":
        columns, data = _parse_json_dict(file)
        return sanitize_varnames(columns)
    elif ext == ".csv":
        df = pd.read_csv(file)
        return sanitize_varnames(df.columns.tolist())
    else:
        raise ValueError(f"File {file} is not a DAT / TXT / JSON / CSV file")


def load_data(
    files: str | Path | List[Path],
) -> pd.DataFrame:
    dfs = []
    if isinstance(files, (str, Path)):
        files = [Path(files)]

    for file in files:
        file = Path(file) if isinstance(file, str) else file
        assert file.exists(), f"File {file} does not exist"
        _, ext = os.path.splitext(file)
        if ext == ".dat" or ext == ".txt":
            lines = _parse_data(file)
            dfs.append(pd.DataFrame(lines[1:], columns=sanitize_varnames(lines[0])))
        elif ext == ".json":
            columns, data = _parse_json_dict(file)
            dfs.append(pd.DataFrame(data=data, columns=sanitize_varnames(columns)))
        elif ext == ".csv":
            dfs.append(pd.read_csv(file))
        else:
            raise ValueError(f"File {file} is not a DAT / TXT / JSON / CSV file")

    df = pd.concat(dfs, ignore_index=True)
    df = sanitize_varnames_df(df)
    return df


def process_input_file(
    files: str | Path | List[Path],
    columns_to_keep: Optional[List[str]] = None,
    columns_to_remove: List[str] = ["interface"],
    make_log: Optional[bool | List[str]] = None,
    custom_operations: Optional[Callable] = None,
    suffix: str = "processed",
    **kwargs,
) -> Path:
    """
    Processes the input file(s) by performing various data manipulation tasks.
    Additional kwargs are passed to _filter_data for data trimming / filtering.

    Args:
        files (str | Path | List[Path]): Path(s) to the input file(s).
        columns_to_remove: List of column names to remove from the dataframe. Defaults to ["interface"].
        make_log: If True, apply logarithmic transformation to all columns (except '%eval_id').
                    If a list of column names is provided, apply the transformation only to those columns. Defaults to None (e.g. no transformation).

    Returns:
        output_file: Path to the processed file.
    """
    if isinstance(files, (str, Path)):
        files = [Path(files)]
    df = load_data(files)
    df = _filter_data(df, **kwargs)
    

    if custom_operations:
        df: pd.DataFrame = custom_operations(df)
        
    if columns_to_keep:
        columns_to_keep = [sanitize_varname(c) for c in columns_to_keep if c in df.columns or sanitize_varname(c) in df.columns]
        df.columns = [sanitize_varname(col) for col in df.columns]
        df = df[columns_to_keep]
    else:
        for c in columns_to_remove:
            c_sanitized = sanitize_varname(c)
            if c_sanitized in df.columns:
                df.drop(c_sanitized, axis=1, inplace=True)
            else:
                print(f"Column {c} (to be removed) not found in the dataframe")

    # Replace spaces and special chars in column names
    df.columns = [sanitize_varname(col) for col in df.columns]
    
    if r"%eval_id" in df.columns:
        df[r"%eval_id"] = np.arange(1, len(df) + 1)

    if make_log:
        log_vars = make_log if isinstance(make_log, list) else df.columns
        for var in log_vars:
            if var != r"%eval_id":
                df[var] = np.log(df[var])
                df.rename(columns={var: "log_" + var}, inplace=True)

    processed_file = Path(
        "_".join([os.path.splitext(f)[0] for f in files]+[suffix]) + ".txt"
    )
    df.to_csv(processed_file, sep=" ", index=False)
    return processed_file


def _filter_data(
    df: pd.DataFrame,
    keep_idxs: Optional[List[int]] = None,
    filter_N_samples: Optional[int] = None,
    filter_highest_N: Optional[int] = None,
    filter_highest_N_variable: Optional[str] = None,
) -> pd.DataFrame:
    """
    The following arguments are optional and can be used to select specific parts of the data.

    - keep_idxs: List of row indices to keep in the dataframe. Defaults to None (all kept)
    - filter_N_samples: Number of rows to keep from the top of the dataframe. Defaults to None.
    - filter_highest_N: Number of rows to keep based on the highest values of a specified column
                    (given by 'filter_highest_N_variable'). Defaults to None.
    """
    ###################### Filtering ###################################
    if filter_highest_N is not None:
        assert (
            filter_N_samples is None
        ), "only one of 'filter_highest_N' or 'filter_N_samples' is allowed"
        filter_highest_N_variable = (
            df.columns[-1]
            if filter_highest_N_variable is None
            else filter_highest_N_variable
        )
        df = df.sort_values(by=filter_highest_N_variable, ascending=False).iloc[
            filter_highest_N:
        ]

    # allows to only take the first N rows
    if filter_N_samples is not None:
        assert (
            filter_highest_N is None
        ), "only one of 'filter_highest_N' or 'filter_N_samples' is allowed"
        df = df.iloc[:filter_N_samples] if filter_N_samples is not None else df
    ## allow to only keep certain idxs
    if keep_idxs is not None:
        original_len = len(df)
        df = df.loc[keep_idxs]
        print(f"Keeping only {len(df)} rows (of {original_len})")

    return df


def get_results(file: Path, key: str = "-AFpeak") -> np.ndarray:
    df = load_data(file)
    results = df[key].values
    results = [float(r) for r in results]
    return np.array(results)


def create_samples_along_axes(
    run_dir: Path,
    data: pd.DataFrame,
    input_vars: List[str],
    NSAMPLESPERVAR: int,
    cut_values: Optional[Dict[str, float]] = None,
    sweep_file_name: str = "sweep_input",
) -> Path:
    # create sweeps data
    if not sweep_file_name.endswith(".csv"):
        sweep_file_name += ".csv"
    SWEEP_INPUT_FILE = run_dir / sweep_file_name
    
    data = sanitize_varnames_df(data)
    input_vars = sanitize_varnames(input_vars)

    assert np.all(
        [var in data.columns for var in input_vars]
    ), "Input variables not found in data"
    data = data[input_vars]
    assert len(data.columns) == len(
        input_vars
    ), "Data columns do not match input variables"
    mins, maxs = data.min().values, data.max().values
    cut_value_list = [cut_values[var] for var in input_vars] if cut_values else data.mean().values

    sample_list = []
    for i, var in enumerate(input_vars):
        sample = copy.deepcopy(cut_value_list)
        ## use avg values for all variables but for the target one
        for j in range(NSAMPLESPERVAR):
            val = mins[i] + (maxs[i] - mins[i]) * j / (NSAMPLESPERVAR - 1)
            sample[i] = val
            sample_list.append({k: v for k, v in zip(input_vars, sample)})
    sweep_df = pd.DataFrame(sample_list)
    sweep_df.to_csv(SWEEP_INPUT_FILE, index=False)
    PROCESSED_SWEEP_INPUT_FILE = Path(process_input_file(SWEEP_INPUT_FILE))

    return PROCESSED_SWEEP_INPUT_FILE


def extract_predictions_along_axes(
    run_dir: Path, RESPONSE: str, input_vars: List[str], NSAMPLESPERVAR: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    To retrieve results generated with 'create_samples_along_axes'
    For a RESPONSE output variable, return a dictionary with input variables as keys.
    Each contains a dictionary with keys "x" (values along the input axes), "y_hat" (predicted values)
    and, if the SuMo provides it, "std_hat", i.e. the sqrt of predicted variance at sample points
    """
    y_hat = get_results(run_dir / "predictions.dat", RESPONSE)
    if (run_dir / "variances.dat").is_file():
        std_hat = np.sqrt(
            get_results(run_dir / "variances.dat", RESPONSE + "_variance")
        )
    input_vars = sanitize_varnames(input_vars)

    results = {}
    for i, variable in enumerate(input_vars):
        x = get_results(run_dir / "predictions.dat", variable)
        results[variable] = {
            "x": list(x[i * NSAMPLESPERVAR : (i + 1) * NSAMPLESPERVAR]),
            "y_hat": list(y_hat[i * NSAMPLESPERVAR : (i + 1) * NSAMPLESPERVAR]),
        }
        if (run_dir / "variances.dat").is_file():
            results[variable].update(
                {"std_hat": list(std_hat[i * NSAMPLESPERVAR : (i + 1) * NSAMPLESPERVAR])}
            )

    return results

def create_grid_samples(
    run_dir: Path,
    grid_vars: List[str],
    input_vars: List[str],
    mins: List[float],
    cut_values: List[float],
    maxs: List[float],
    n_points_per_dimension: List[int],
    downscaling_factor: Optional[float] = None,
    gridpoints_file_name: str = "grid_input.csv",
) -> Path:
    """Generate grid points (either for sampling, or to evaluate the SuMo upon and display)"""
    GRIDPOINTS_INPUT_FILE = run_dir / gridpoints_file_name
    if len(input_vars) != len(n_points_per_dimension):
        raise ValueError(
            "Number of variables must match number of points per dimension."
        )
    if len(input_vars) != len(mins):
        raise ValueError("Number of variables must match number of mins.")
    if len(input_vars) != len(maxs):
        raise ValueError("Number of variables must match number of maxs.")
    if len(input_vars) < 1:
        raise ValueError("At least one variable is required to generate a grid.")
    
    if downscaling_factor is not None:
        n_points_per_dimension = [
            int(np.ceil(n / downscaling_factor)) for n in n_points_per_dimension
        ]
    
    input_vars = sanitize_varnames(input_vars)
    grid_vars = sanitize_varnames(grid_vars)
    
    print("Parameters to create grid: ")
    print(mins, maxs, n_points_per_dimension)
    print("Grid vars: ", grid_vars)
    print("input vars: ", input_vars)
    
    grid = np.meshgrid(
        *[
            np.linspace(mins[i], maxs[i], n_points_per_dimension[i])
            if input_vars[i] in grid_vars
            else cut_values[i]
            for i in range(len(n_points_per_dimension))
        ]
    )
    gridpoints = np.vstack([a.ravel() for a in grid]).T
    gridpoints = pd.DataFrame(gridpoints, columns=input_vars)
    
    gridpoints.to_csv(GRIDPOINTS_INPUT_FILE, index=False)
    PROCESSED_GRIDPOINTS_INPUT_FILE = Path(process_input_file(GRIDPOINTS_INPUT_FILE))

    return PROCESSED_GRIDPOINTS_INPUT_FILE


def extract_predictions_gridpoints(
    run_dir: Path, RESPONSE: str, input_vars: List[str], NSAMPLESPERVAR: int
) -> Dict[str, List[float]]:
    """
    To retrieve results generated with 'create_samples_along_axes'
    For a RESPONSE output variable, return a dictionary with input variables as keys.
    Each contains a dictionary with keys "x" (values along the input axes), "y_hat" (predicted values)
    and, if the SuMo provides it, "std_hat", i.e. the sqrt of predicted variance at sample points
    """
    predictions_df = load_data(run_dir / "predictions.dat")
    input_vars = sanitize_varnames(input_vars)
    RESPONSE = sanitize_varname(RESPONSE)
    
    y_hat = get_results(run_dir / "predictions.dat", RESPONSE)

    results = {var: predictions_df[var].astype(float).tolist() for var in input_vars}
    results[RESPONSE] = y_hat.astype(float).tolist()
    if (run_dir / "variances.dat").is_file():
        std_hat = np.sqrt(
            get_results(run_dir / "variances.dat", RESPONSE + "_variance")
        )
        results[RESPONSE + "_std"] = std_hat.astype(float).tolist() # type: ignore

    return results

def create_manual_uq_samples(input_vars: List[str], distributions: Dict[str, Dict[str, float]], num_samples: int, seed: Optional[int] = None):
    """
    Generate samples for manual UQ propagation based on user-specified distributions.
    Returns a list of dictionaries, each representing a sample.
    """
    input_vars = sanitize_varnames(input_vars)
    distributions = {sanitize_varname(k): sanitize_varnames_dict(v) for k, v in distributions.items()}
    
    # rng = np.random.default_rng(seed=seed)
    from scipy.stats import norm, uniform
    samples = {}
    for var in input_vars:
        dist_info = distributions[var]
        dist_type = dist_info["distribution"]
        if dist_type == "normal":
            mean = dist_info["mean"]
            std = dist_info["std"]
            samples[var] = norm.rvs(size=num_samples, loc=mean, scale=std).tolist()  # type: ignore
        elif dist_type == "uniform":
            min_val = dist_info["min"]
            max_val = dist_info["max"]
            samples[var] = uniform.rvs(size=num_samples, loc=min_val, scale=max_val-min_val).tolist()  # type: ignore
        elif dist_type == "constant":
            value = dist_info["value"]
            samples[var] = [float(value)] * num_samples
        # elif dist_type == "lognormal":
        #     mean = dist_info["mean"]
        #     std = dist_info["std"]
        #     samples[var] = float(rng.lognormal(mean, std, (num_samples,)))
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")
    return samples


T = TypeVar('T')

@overload
def sanitize_varnames(input_data: str) -> str: ...

@overload
def sanitize_varnames(input_data: List[str]) -> List[str]: ...

@overload
def sanitize_varnames(input_data: Dict[str, T]) -> Dict[str, T]: ...

@overload
def sanitize_varnames(input_data: pd.DataFrame) -> pd.DataFrame: ...

def sanitize_varnames(input_data):
    """
    Sanitize variable names by replacing spaces and non-alphanumeric characters with underscores.
    This function handles different input types:
    - str: sanitizes a single variable name
    - list/iterable: sanitizes each item in the list
    - dict: sanitizes the keys of the dictionary
    - pd.DataFrame: sanitizes the column names
    
    Args:
        input_data: The data to sanitize (string, list, dict, or DataFrame)
        
    Returns:
        Sanitized version of the input data (same type as input)
    """
    # Helper function for sanitizing a single string
    def _sanitize_single(varname: str) -> str:
        # Replace spaces with underscores and then replace any remaining non-alphanumeric chars (except _*-+/)
        return re.sub(r'[^0-9a-zA-Z_*-+/]', '_', varname.replace(' ', '_'))
    
    # Handle different input types
    if isinstance(input_data, str):
        return _sanitize_single(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()  # Create a copy to avoid modifying the original DataFrame
        df.columns = [_sanitize_single(col) for col in df.columns]
        return df
    elif isinstance(input_data, dict):
        # Recursively handle dictionaries
        result = {}
        for k, v in input_data.items():
            sanitized_key = _sanitize_single(k)
            if isinstance(v, dict):
                result[sanitized_key] = sanitize_varnames(v)
            else:
                result[sanitized_key] = v
        return result
    elif hasattr(input_data, '__iter__') and not isinstance(input_data, (str, bytes)):
        return [_sanitize_single(v) for v in input_data]
    else:
        raise TypeError(f"Unsupported input type: {type(input_data)}")

# Aliases for backward compatibility
sanitize_varname = sanitize_varnames  # For single string input
sanitize_varnames_dict = sanitize_varnames  # For dictionary input
sanitize_varnames_df = sanitize_varnames  # For DataFrame input