## TODO make a test script (rather than standalone)
## TODO make much more modular (slowly)
from pathlib import Path
from typing import List
import pandas as pd
import shutil
from utils.dakota_object import DakotaObject
from utils.funs_data_processing import (
    get_variable_names,
    process_input_file,
)
from utils.funs_evaluate import create_run_dir
from utils.funs_create_dakota_conf import create_sumo_evaluation
from utils.funs_plotting import plot_response_curves
from utils.funs_data_processing import (
    create_samples_along_axes,
    extract_predictions_along_axes,
)

################## CONFIG ########################################
##################################################################
TRAINING_FILE = Path("./data/results_Final_50LHS_TitrationProcessed.csv")
run_dir = create_run_dir(Path("."), "evaluate")
N_INPUTS = 5 + 6  # EM + Thermal
N_OUTPUTS = (
    4  #### Current Shunting
    + 4 * 3  #### Dosimetry in 3 areas
    + 3  #### Thermal
    # + 7  #### Neuro
)
##################################################################


def normalize_thermals(df: pd.DataFrame):
    """Custom operations for NIH dataset.
    Normalize Thermal peaks by total current. This normalization is already in the simulation,
    but Thermal deposition scales with power (e.g. square of current) so an additional normalization is necessary.
    """
    for c in df.columns:
        assert (
            "log" not in c
        ), "Log found. The normalization operation assumes variables in linear space."

        if "Thermal_Peak" in c:
            df[c] = df[c] / df["EM_Shunting_Total_Current"]

    return df


##################################################################


# load (and process) data (inc taking logs)
TRAINING_FILE = Path(shutil.copy(TRAINING_FILE, run_dir))
var_names = get_variable_names(TRAINING_FILE)
assert len(var_names) == N_INPUTS + N_OUTPUTS, (
    f"Number of variables in data {len(var_names)} does not coincide with "
    f"expected number of input variables {N_INPUTS} and output responses {N_OUTPUTS}"
)
PROCESSED_TRAINING_FILE = process_input_file(
    TRAINING_FILE, make_log=var_names[:N_INPUTS], custom_operations=normalize_thermals
)
var_names = get_variable_names(PROCESSED_TRAINING_FILE)
input_vars = var_names[:N_INPUTS]
output_vars = var_names[-N_OUTPUTS:]

# create sweeps data
NSAMPLESPERVAR = 21
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
    output_responses=output_vars,
)

# run dakota
dakobj = DakotaObject(
    map_object=None
)  # no need to evaluate any function (only the SuMo, internal to Dakota)
dakobj.run(dakota_conf, run_dir)


# extract output data
def extract_and_plot_prediction_curves(
    run_dir, RESPONSE: str, input_vars: List[str], NSAMPLESPERVAR: int
):
    results = extract_predictions_along_axes(
        run_dir, RESPONSE, input_vars, NSAMPLESPERVAR
    )
    plot_response_curves(results, RESPONSE, input_vars, savedir=run_dir)


extract_and_plot_prediction_curves(run_dir, output_vars[16], input_vars, NSAMPLESPERVAR)

print("DONE")
