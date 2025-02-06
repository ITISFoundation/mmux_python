## TODO make a test script (rather than standalone)
## TODO make much more modular (slowly)
from pathlib import Path
import pandas as pd
import shutil
from utils.funs_data_processing import (
    get_variable_names,
    process_input_file,
)
from utils.funs_evaluate import create_run_dir
from utils.funs_evaluate import evaluate_sumo_along_axes


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
    TRAINING_FILE, make_log=True, custom_operations=normalize_thermals
)
var_names = get_variable_names(PROCESSED_TRAINING_FILE)
input_vars = var_names[:N_INPUTS]
output_vars = var_names[-N_OUTPUTS:]
### FIXME not able to give only some output vars; would need to
# remove them from the (PROCESSED_)TRAINING_FILE as well
PROCESSED_TRAINING_FILE = process_input_file(
    PROCESSED_TRAINING_FILE, columns_to_remove=output_vars[:15]
)

evaluate_sumo_along_axes(run_dir, PROCESSED_TRAINING_FILE, input_vars, output_vars[15:])
## TODO make proper plotting (in log-log scale / labels)
# eventually code all the variaations (e.g. input log or linear; Plot desired in log or linear)
## TODO fix the label titles

print("DONE")
