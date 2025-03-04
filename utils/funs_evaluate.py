from pathlib import Path
import datetime
import os
import pandas as pd
import numpy as np
import shutil
from typing import List, Literal, Optional, Callable, Dict
from utils.dakota_object import DakotaObject
from utils.funs_create_dakota_conf import create_sumo_evaluation, create_uq_propagation
from utils.funs_plotting import plot_response_curves, plot_uq_histogram
import json
from utils.funs_data_processing import (
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
) -> Path:
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
    savepath = plot_response_curves(
        results,
        response_var,
        input_vars,
        savedir=run_dir,
        plotting_xscale=xscale,
        plotting_yscale=yscale,
        label_converter=label_converter,
    )

    assert savepath is not None
    assert savepath.exists(), f"Plotting failed, savepath {savepath} does not exist"
    return savepath


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


if __name__ == "__main__":
    print("DONE")
