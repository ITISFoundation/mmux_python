from typing import List, Optional, Callable, Dict
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import copy


def _parse_data(file: str | Path) -> List[List[str]]:
    data = []
    with open(file) as f:
        data = [line.strip().split() for line in f]
    return data


def _parse_json_dict(file: str | Path):
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
        return lines[0]
    elif ext == ".json":
        columns, data = _parse_json_dict(file)
        return columns
    elif ext == ".csv":
        df = pd.read_csv(file)
        return df.columns.tolist()
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
            dfs.append(pd.DataFrame(lines[1:], columns=lines[0]))
        elif ext == ".json":
            columns, data = _parse_json_dict(file)
            dfs.append(pd.DataFrame(data=data, columns=columns))
        elif ext == ".csv":
            dfs.append(pd.read_csv(file))
        else:
            raise ValueError(f"File {file} is not a DAT / TXT / JSON / CSV file")

    df = pd.concat(dfs, ignore_index=True)
    return df


def process_input_file(
    files: str | Path | List[Path],
    columns_to_keep: Optional[List[str]] = None,
    columns_to_remove: List[str] = ["interface"],
    make_log: Optional[bool | List[str]] = None,
    custom_operations: Optional[Callable] = None,
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
        columns_to_keep = [c for c in columns_to_keep if c in df.columns]
        df = df[columns_to_keep]
    else:
        for c in columns_to_remove:
            if c in df.columns:
                df.drop(c, axis=1, inplace=True)
            else:
                print(f"Column {c} (to be removed) not found in the dataframe")

    if r"%eval_id" in df.columns:
        df[r"%eval_id"] = np.arange(1, len(df) + 1)

    if make_log:
        log_vars = make_log if isinstance(make_log, list) else df.columns
        for var in log_vars:
            if var != r"%eval_id":
                df[var] = np.log(df[var])
                df.rename(columns={var: "log_" + var}, inplace=True)

    processed_file = Path(
        "_".join([os.path.splitext(f)[0] for f in files]) + "_processed.txt"
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
        df = df[keep_idxs]
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
    sweep_file_name: str = "sweep_input",
) -> Path:
    # create sweeps data
    if not sweep_file_name.endswith(".csv"):
        sweep_file_name += ".csv"
    SWEEP_INPUT_FILE = run_dir / sweep_file_name

    assert np.all(
        [var in data.columns for var in input_vars]
    ), "Input variables not found in data"
    data = data[input_vars]
    assert len(data.columns) == len(
        input_vars
    ), "Data columns do not match input variables"
    avgs, mins, maxs = data.mean().values, data.min().values, data.max().values

    sample_list = []
    for i, var in enumerate(input_vars):
        sample = copy.deepcopy(avgs)
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
    run_dir: Path, RESPONSE: str, input_vars: List[str], NSAMPLESPERVAR: int
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
