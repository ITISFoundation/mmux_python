from pathlib import Path
import datetime
import os
from typing import List
from utils.dakota_object import DakotaObject
from utils.funs_create_dakota_conf import create_sumo_evaluation
from utils.funs_plotting import plot_response_curves
from utils.funs_data_processing import (
    create_samples_along_axes,
    extract_predictions_along_axes,
)
import pandas as pd


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
    response_vars: List[str],
    NSAMPLESPERVAR: int = 21,
    ## TODO be able to load / query SuMo directly; or simply be able to do on any function (although prob better as separate function, that)
):
    # create sweeps data
    data = pd.read_csv(PROCESSED_TRAINING_FILE, sep=" ")
    PROCESSED_SWEEP_INPUT_FILE = create_samples_along_axes(
        run_dir, data, input_vars, NSAMPLESPERVAR
    )

    # create dakota file
    dakota_conf = create_sumo_evaluation(
        build_file=PROCESSED_TRAINING_FILE,
        ### TODO be able to save & load surrogate models (start w GP) rather than create them every time
        samples_file=PROCESSED_SWEEP_INPUT_FILE,
        input_variables=input_vars,
        output_responses=response_vars,
    )

    # run dakota
    dakobj = DakotaObject(
        map_object=None
    )  # no need to evaluate any function (only the SuMo, internal to Dakota)
    dakobj.run(dakota_conf, run_dir)

    for RESPONSE in response_vars:
        results = extract_predictions_along_axes(
            run_dir, RESPONSE, input_vars, NSAMPLESPERVAR
        )
        plot_response_curves(results, RESPONSE, input_vars, savedir=run_dir)


if __name__ == "__main__":
    print("DONE")
