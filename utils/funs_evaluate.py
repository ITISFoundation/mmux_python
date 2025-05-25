from pathlib import Path
import datetime
import os
import pandas as pd
import numpy as np
import shutil
import json
from typing import List, Literal, Optional, Callable, Dict

from mmux_python.utils.dakota_object import DakotaObject
from mmux_python.utils.funs_create_dakota_conf import create_sumo_evaluation, create_uq_propagation, create_sumo_crossvalidation
from mmux_python.utils.funs_plotting import plot_response_curves, plot_uq_histogram
from mmux_python.utils.funs_data_processing import (
    create_samples_along_axes,
    extract_predictions_along_axes,
    get_results,
)


def create_run_dir(script_dir: Path, dir_name: str = "sampling"):
    ## part 1 - setup
    main_runs_dir = script_dir / "runs"
    current_time = datetime.datetime.now().strftime("%Y%m%d.%H%M%S%d")
    temp_dir = main_runs_dir / "_".join(["dakota", current_time, dir_name])
    print(str(temp_dir))
    os.makedirs(temp_dir, exist_ok=True)
    print("temp_dir: ", temp_dir)
    return temp_dir

def retrieve_csv_result(
    csv_file_path: str, inputs: Dict[str, float], outputs: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Retrieve the result from a csv file.
    """

    df = pd.read_csv(csv_file_path)

    for col in inputs.keys():
        if col not in df.columns:
            raise ValueError(
                f"Input {col} not in the csv file. Columns are: {df.columns.values}"
            )
            
    if outputs is not None:
        for col in outputs:
            if col not in df.columns:
                raise ValueError(
                    f"Output {col} not in the csv file. Columns are: {df.columns.values}"
                )
        result = df.loc[np.all(df[inputs.keys()] == inputs.values(), axis=1), outputs]
    else:
        result = df.loc[np.all(df[inputs.keys()] == inputs.values(), axis=1)]
    # Check if the result is empty or has multiple rows
    assert len(result) != 0, f"No result found for inputs {inputs}."
    assert len(result) == 1, f"Multiple results found for inputs {inputs}."

    return result.iloc[0].to_dict()


def evaluate_sumo_along_axes(
    run_dir: Path,
    PROCESSED_TRAINING_FILE: Path,
    input_vars: List[str],
    response_var: str,
    sumo_import_name: Optional[str] = None,
    sumo_export_name: Optional[str] = None,
    NSAMPLESPERVAR: int = 21,
    xscale: Literal["linear", "log"] = "linear",
    yscale: Literal["linear", "log"] = "linear",
    label_converter: Optional[Callable] = None,
    MAKEPLOT: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Given a training data to create a SuMo, generate it, and plot the profile along the central axes
    (e.g. all variables but the sweeped one will be set to its central value).
    No callback is necessary (everything internal to Dakota).

    Log / Linear scale of the variable is inferred its name; mean value is taken in the corresponding scale.
    Plots scales (after SuMo creation and sampling) can be either linear or logarithmic.
    """
    # create sweeps data
    data = pd.read_csv(PROCESSED_TRAINING_FILE, sep=" ")
    PROCESSED_SWEEP_INPUT_FILE = create_samples_along_axes(
        run_dir, data, input_vars, NSAMPLESPERVAR
    )

    if sumo_import_name:
        models_dir = run_dir.parent / "models"
        assert (
            models_dir.exists()
        ), f"Models dir {models_dir} does not exist, but SuMo import is trying to copy files there"
        for file in models_dir.glob(f"{sumo_import_name}*"):
            shutil.copy(file, run_dir)

    # create dakota file
    dakota_conf = create_sumo_evaluation(
        build_file=PROCESSED_TRAINING_FILE,
        sumo_import_name=sumo_import_name,
        sumo_export_name=sumo_export_name,
        ### TODO once this works, try to get it to work wo evaluation (or just one sample, if not possible?)
        samples_file=PROCESSED_SWEEP_INPUT_FILE,
        input_variables=input_vars,
        output_responses=[response_var],
    )

    # run dakota
    dakobj = DakotaObject(
        map_object=None
    )  # no need to evaluate any function (only the SuMo, internal to Dakota)
    dakobj.run(dakota_conf, run_dir)

    if sumo_export_name:
        models_dir = run_dir.parent / "models"
        os.makedirs(models_dir, exist_ok=True)
        for file in run_dir.glob(f"{sumo_export_name}*"):
            shutil.copy(file, models_dir)
            # Also save input and output variables to a JSON file
            json_data = {"input_vars": input_vars, "output_var": response_var}
            json_save_path = models_dir / f"{file.name}.json"
            with open(json_save_path, "w") as json_file:
                json.dump(json_data, json_file, indent=4)

    results = extract_predictions_along_axes(
        run_dir, response_var, input_vars, NSAMPLESPERVAR
    )

    if MAKEPLOT:
        plot_response_curves(
            results,
            response_var,
            input_vars,
            savedir=run_dir,
            plotting_xscale=xscale,
            plotting_yscale=yscale,
            label_converter=label_converter,
        )

    return results

### TODO refactor in new MMUX-compatible version (like above)
def propagate_uq(
    run_dir: Path,
    PROCESSED_TRAINING_FILE: Path,
    input_vars: List[str],
    output_response: str,
    means: Dict[str, float],
    stds: Dict[str, float],
    xscale: Literal["linear", "log"] = "linear",
    label_converter: Optional[Callable] = None,
):

    # create dakota file
    dakota_conf = create_uq_propagation(
        build_file=PROCESSED_TRAINING_FILE,
        input_variables=input_vars,
        input_means=means,
        input_stds=stds,
        output_responses=[output_response],
    )

    # run dakota
    dakobj = DakotaObject(
        map_object=None
    )  # no need to evaluate any function (only the SuMo, internal to Dakota)
    dakobj.run(dakota_conf, run_dir)
    x = get_results(run_dir / f"predictions.dat", output_response)
    savepath = plot_uq_histogram(
        x,
        output_response,
        savedir=run_dir,
        plotting_xscale=xscale,
        label_converter=label_converter,
    )

    return savepath

def _parse_crossvalidation_outputlogs(log_output: str, N_CROSS_VALIDATION: int):
    import re

    variable_name_pattern = (
        f"Surrogate quality metrics \({N_CROSS_VALIDATION}-fold CV\) for (\w+):"
    )
    metrics_pattern = r"\s+(root_mean_squared|sum_abs|mean_abs|max_abs)\s+([\d.e+-]+|nan)"


    # Find all occurrences of variable names in the log
    variables = re.findall(variable_name_pattern, log_output)

    # Split the log output by the variable name to handle each output separately
    log_parts = re.split(variable_name_pattern, log_output)
    log_parts = log_parts[1:]  # Skip the first part (before the first variable name)

    # Dictionary to hold the parsed results for each output variable
    parsed_error_metrics = {}

    # Loop through the log parts, and extract metrics for each output variable
    for i, variable in enumerate(variables):
        # The log part after each variable name contains the metrics section for that variable
        metrics_section = log_parts[
            2 * i + 1
        ]  # The log part immediately after the variable name

        ## remove the training error of the next variable
        metrics_section = metrics_section.split("build (training) points")[0]

        # Find all the surrogate quality metrics for this particular output variable
        metrics_matches = re.findall(metrics_pattern, metrics_section)

        if metrics_matches:
            metrics = {metric: value for metric, value in metrics_matches}
            parsed_error_metrics[variable] = metrics
        else:
            parsed_error_metrics[variable] = "No surrogate quality metrics found."

    print(parsed_error_metrics)
    return parsed_error_metrics


def evaluate_sumo_crossvalidation(
    run_dir: Path,
    PROCESSED_TRAINING_FILE: Path,
    input_vars: List[str],
    output_response: str,
    N_CROSS_VALIDATION: int = 5,
):
    dakota_conf = create_sumo_crossvalidation(
        PROCESSED_TRAINING_FILE, 
        input_vars,
        [output_response],
        N_CROSS_VALIDATION=N_CROSS_VALIDATION,
    )
        # run dakota
    dakobj = DakotaObject(
        map_object=None
    )  # no need to evaluate any function (only the SuMo, internal to Dakota)
    dakobj.run(dakota_conf, run_dir)
    ## TODO I was parsing from the stdout. How to do it now?
    log_output = ""
    parsed_error_metrics = _parse_crossvalidation_outputlogs(log_output, 
                                                             N_CROSS_VALIDATION)
    
    return parsed_error_metrics 

if __name__ == "__main__":
    print("DONE")
