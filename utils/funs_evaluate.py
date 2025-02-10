from pathlib import Path
import datetime
import os
from typing import List, Literal, Optional, Callable
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
    ## TODO be able to load / query SuMo directly; or simply be able to do on any function (although prob better as separate function, that)
    plotting_input_vars: List[str],
    response_vars: List[str],
    NSAMPLESPERVAR: int = 21,
    xscale: Literal["linear", "log"] = "linear",
    yscale: Literal["linear", "log"] = "linear",
    label_converter: Optional[Callable] = None,
):
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
        results = {k: v for k, v in results.items() if k in plotting_input_vars}
        plot_response_curves(
            results,
            RESPONSE,
            plotting_input_vars,
            savedir=run_dir,
            plotting_xscale=xscale,
            plotting_yscale=yscale,
            label_converter=label_converter,
        )


if __name__ == "__main__":
    print("DONE")
