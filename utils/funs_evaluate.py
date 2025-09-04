from pathlib import Path
import datetime
import os
import re
import uuid
import pandas as pd
import numpy as np # type: ignore
import shutil
from typing import List, Literal, Optional, Callable, Dict
from sklearn.model_selection import KFold

import sys
sys.path.append(str(Path(__file__).parent))
from dakota_object import DakotaObject
from funs_create_dakota_conf import (
    create_sumo_evaluation_conffile, 
    create_uq_propagation_conffile, 
    create_sumo_crossvalidation_conffile, 
    create_sumo_manual_crossvalidation_conffile, 
    create_moga_optimization_conffile,
)
from funs_data_processing import (
    sanitize_varnames,
    create_samples_along_axes,
    extract_predictions_along_axes,
    create_grid_samples,
    extract_predictions_gridpoints,
    get_results,
    load_data,
    get_non_dominated_indices,
    get_bounds_uniform_distributions
)


def create_run_dir(script_dir: Path, dir_name: str = "sampling"):
    ## part 1 - setup
    main_runs_dir = script_dir / "runs"
    current_time = datetime.datetime.now().strftime("%Y%m%d.%H%M%S%d")
    uid = uuid.uuid4().hex
    temp_dir = main_runs_dir / "_".join(["dakota", current_time, uid, dir_name])
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
    cut_values: Optional[Dict[str, float]] = None,
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
    # sanitize variable names
    input_vars = sanitize_varnames(input_vars)
    response_var = sanitize_varnames(response_var)
    cut_values = sanitize_varnames(cut_values) if cut_values else None
    
    # create sweeps data
    data = pd.read_csv(PROCESSED_TRAINING_FILE, sep=" ")
    PROCESSED_SWEEP_INPUT_FILE = create_samples_along_axes(
        run_dir, data, input_vars, NSAMPLESPERVAR, cut_values=cut_values
    )

    if sumo_import_name:
        models_dir = run_dir.parent / "models"
        assert (
            models_dir.exists()
        ), f"Models dir {models_dir} does not exist, but SuMo import is trying to copy files there"
        for file in models_dir.glob(f"{sumo_import_name}*"):
            shutil.copy(file, run_dir)

    # create dakota file
    dakota_conf = create_sumo_evaluation_conffile(
        build_file=PROCESSED_TRAINING_FILE,
        sumo_import_name=sumo_import_name,
        sumo_export_name=sumo_export_name,
        samples_file=PROCESSED_SWEEP_INPUT_FILE,
        input_variables=input_vars,
        output_responses=[response_var],
    )

    # run dakota
    dakobj = DakotaObject()
    dakobj.run(dakota_conf, run_dir)
    results = extract_predictions_along_axes(
        run_dir, response_var, input_vars, NSAMPLESPERVAR
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
    n_samples: int = 1000,
    xscale: Literal["linear", "log"] = "linear",
    label_converter: Optional[Callable] = None,
) -> List[float]:
    input_vars = sanitize_varnames(input_vars)
    output_response = sanitize_varnames(output_response)
    means = {sanitize_varnames(k): v for k, v in means.items()}
    stds = {sanitize_varnames(k): v for k, v in stds.items()}

    # create dakota file
    dakota_conf = create_uq_propagation_conffile(
        build_file=PROCESSED_TRAINING_FILE,
        input_variables=input_vars,
        input_means=means,
        input_stds=stds,
        output_responses=[output_response],
        n_samples=n_samples,
    )

    # run dakota
    dakobj = DakotaObject()
    dakobj.run(dakota_conf, run_dir)
    x = get_results(run_dir / f"predictions.dat", output_response)
    return x.tolist()

def _parse_crossvalidation_outputlogs(log_output: str, N_CROSS_VALIDATION: int):

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
    input_vars = sanitize_varnames(input_vars)
    output_response = sanitize_varnames(output_response)

    dakota_conf = create_sumo_crossvalidation_conffile(
        PROCESSED_TRAINING_FILE, 
        input_vars,
        [output_response],
        N_CROSS_VALIDATION=N_CROSS_VALIDATION,
    )
        # run dakota
    dakobj = DakotaObject()
    dakobj.run(dakota_conf, run_dir)
    ## TODO I was parsing from the stdout. How to do it now?
    log_output = ""
    parsed_error_metrics = _parse_crossvalidation_outputlogs(log_output, 
                                                             N_CROSS_VALIDATION)
    
    return parsed_error_metrics 

def evaluate_sumo_manual_crossvalidation(
    run_dir: Path,
    PROCESSED_TRAINING_FILE: Path,
    input_vars: List[str],
    output_response: str,
    N_CROSS_VALIDATION: int = 5,
):
    input_vars = sanitize_varnames(input_vars)
    output_response = sanitize_varnames(output_response)

    all_observations = load_data(PROCESSED_TRAINING_FILE)[output_response].astype(float)
    n_samples = len(all_observations)
    indices = np.arange(n_samples)
    all_predictions = np.empty(n_samples)
    all_stds = np.empty(n_samples)
    kf = KFold(n_splits=N_CROSS_VALIDATION, shuffle=True, random_state=42)

    for fold, (_, val_idx) in enumerate(kf.split(indices)):
        fold_run_dir = run_dir / f"fold_{fold}"
        os.makedirs(fold_run_dir, exist_ok=True)

        # Create Dakota config for this fold
        dakota_conf = create_sumo_manual_crossvalidation_conffile(
            fold_run_dir,
            PROCESSED_TRAINING_FILE,
            input_vars,
            output_response,
            validation_indices=val_idx.tolist(),
            dakota_conf_file= fold_run_dir / "dakota_config.in",
        )
        dakobj = DakotaObject()
        dakobj.run(dakota_conf, fold_run_dir)

        # Extract predictions for this fold and store in the correct positions
        fold_predictions = get_results(fold_run_dir / "predictions.dat", output_response)
        print(f"Fold {fold} predictions: {fold_predictions}")
        print(f"Validation indices: {val_idx}")
        
        all_predictions[val_idx] = fold_predictions
        if (fold_run_dir / "variances.dat").is_file():
            fold_var = get_results(fold_run_dir / "variances.dat", output_response+"_variance")
            all_stds[val_idx] = np.sqrt(fold_var)

    return {
        output_response: all_observations.tolist(),
        output_response + "_hat": all_predictions.tolist(),
        output_response + "_std_hat": all_stds.tolist(),
    }

def evaluate_sumo(
    run_dir: Path, 
    PROCESSED_TRAINING_FILE: Path,
    PROCESSED_EVALUATION_SAMPLES_FILE: Path,
    input_vars: List[str],
    response_var: str,
) -> Dict[str, List[float]]:
    input_vars = sanitize_varnames(input_vars)
    response_var = sanitize_varnames(response_var)

    """Given a training data to create a SuMo, generate it, and evaluate on the training data.
    No callback is necessary (everything internal to Dakota).
    """
    # create dakota file
    dakota_conf = create_sumo_evaluation_conffile(
        build_file=PROCESSED_TRAINING_FILE,
        samples_file=PROCESSED_EVALUATION_SAMPLES_FILE,
        input_variables=input_vars,
        output_responses=[response_var],
    )

    # run dakota
    dakobj = DakotaObject()
    dakobj.run(dakota_conf, run_dir)

    results = {response_var+"_hat": get_results(run_dir / "predictions.dat", response_var).tolist()}
    if (run_dir / "variances.dat").is_file():
        variances = get_results(run_dir / "variances.dat", response_var + "_variance")
        results[response_var + "_std_hat"] = np.sqrt(variances).tolist()

    return results

def evaluate_sumo_on_grid(
    run_dir: Path,
    PROCESSED_TRAINING_FILE: Path,
    grid_vars: List[str],
    input_vars: List[str],
    response_var: str,
    cut_values: Optional[Dict[str, float]] = None,
    # sumo_import_name: Optional[str] = None,
    # sumo_export_name: Optional[str] = None,
    NSAMPLESPERVAR: int = 21,
    # xscale: Literal["linear", "log"] = "linear",
    # yscale: Literal["linear", "log"] = "linear",
    # label_converter: Optional[Callable] = None,
    # MAKEPLOT: bool = False,
) -> Dict[str, List[float]]:
    """Given a training data to create a SuMo, generate it, and evaluate on a grid of points.
    The grid is created by sweeping the variables in `grid_vars` over their min and max values,
    while the other variables in `input_vars` are set to their central values.
    The grid is created by sampling `NSAMPLESPERVAR` points per variable.
    The results are returned as a dictionary, where the keys are the variable names and the values are lists of values (inputs / predictions).
    No callback is necessary (everything internal to Dakota).

    Log / Linear scale of the variable is inferred its name; mean value is taken in the corresponding scale.
    Plots scales (after SuMo creation and sampling) can be either linear or logarithmic.
    """
    NPOINTSPERDIMENSION = [NSAMPLESPERVAR] * len(input_vars)  # default number of points per dimension
    grid_vars = sanitize_varnames(grid_vars)
    input_vars = sanitize_varnames(input_vars)
    response_var = sanitize_varnames(response_var)
    cut_values = sanitize_varnames(cut_values) if cut_values else None

    # create sweeps data
    data = pd.read_csv(PROCESSED_TRAINING_FILE, sep=" ")
    PROCESSED_GRIDPOINTS_INPUT_FILE = create_grid_samples(
        run_dir = run_dir,
        grid_vars = grid_vars,
        input_vars = input_vars,
        mins = [data[var].min() for var in input_vars], ## TODO it is here that we should use the distribution values (passed directly from the frontend)
        cut_values = [cut_values[var] for var in input_vars] if cut_values else [data[var].mean() for var in input_vars],
        maxs = [data[var].max() for var in input_vars], # TODO it is here that we should use the distribution values (passed directly from the frontend)
        n_points_per_dimension=NPOINTSPERDIMENSION,
    )

    # create dakota file
    dakota_conf = create_sumo_evaluation_conffile(
        build_file=PROCESSED_TRAINING_FILE,
        # sumo_import_name=sumo_import_name,
        # sumo_export_name=sumo_export_name,
        ### TODO once this works, try to get it to work wo evaluation (or just one sample, if not possible?)
        samples_file=PROCESSED_GRIDPOINTS_INPUT_FILE,
        input_variables=input_vars,
        output_responses=[response_var],
    )

    dakobj = DakotaObject()
    dakobj.run(dakota_conf, run_dir)

    results = extract_predictions_gridpoints(
        run_dir, response_var, input_vars, NSAMPLESPERVAR
    )
    
    if len(grid_vars) == 2: ## this is not necessary for 3D
        output = np.array(results[response_var])
        reshape_indices = [NPOINTSPERDIMENSION[i] for i in range(len(input_vars)) if input_vars[i] in grid_vars ]
        if grid_vars[0] in input_vars[:2] and grid_vars[1] in input_vars[:2]:
            ## reshape fills in row order. For some reason, this needs to be done reversed in XY / YX cases
            ## but NOT for any other input combination...
            output = output.reshape(reshape_indices[::-1]).T  
        else:
            output = output.reshape(reshape_indices)
        input_vars_in_grid_vars = [var for var in input_vars if var in grid_vars ]
        transpose_indices = [input_vars_in_grid_vars.index(grid_vars[i]) for i in range(len(grid_vars))]
        final_output = output.transpose(transpose_indices[::-1]) # ZX, XZ, YZ, ZY work; but not YX, XY. Why??? 
        results[response_var] = final_output.tolist()
        
    return results

def perform_moga_optimization(
    run_dir: Path,
    PROCESSED_TRAINING_FILE: Path,
    input_vars: List[str],
    distributions: Dict[str, Dict[str, float]],
    output_responses: List[str],
    moga_kwargs: dict,
) -> Dict[str, List[float | int]]:
    print(f"minimizing {', '.join(output_responses)}")

    input_vars = sanitize_varnames(input_vars)
    output_responses = [sanitize_varnames(resp) for resp in output_responses]
    distributions = sanitize_varnames(distributions)

    # assumes uniform distribution for MOGA - raises Error otherwise
    lower_bounds, upper_bounds = get_bounds_uniform_distributions(input_vars, distributions)

    # create dakota file
    dakota_conf = create_moga_optimization_conffile(
        build_file=PROCESSED_TRAINING_FILE,
        input_variables=input_vars,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        output_responses=output_responses,
        moga_kwargs=moga_kwargs,
        dakota_conf_file=run_dir / "dakota_config.in",
    )

    # run dakota
    dakobj = DakotaObject()
    dakobj.run(dakota_conf, run_dir)

    results = {}
    for res in output_responses:
        x = get_results(run_dir / f"predictions.dat", res)
        results[res] = x.tolist()
    for inv in input_vars:
        x = get_results(run_dir / f"predictions.dat", inv)
        results[inv] = x.tolist()

    results_df = load_data(run_dir / "results.dat")
    non_dominated_indices = get_non_dominated_indices(
        results_df,
        optimized_vars=output_responses[:2],
        sort_by_column=output_responses[0],
        optimization_modes=None, ## minimization by default
    )
    results["non_dominated_indices"] = np.array(non_dominated_indices).astype(float).tolist() ## int64 is not JSON serializable

    return results


if __name__ == "__main__":
    print("DONE")
