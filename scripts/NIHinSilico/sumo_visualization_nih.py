from pathlib import Path
from typing import Callable
import shutil
import sys

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.funs_data_processing import (
    get_variable_names,
    process_input_file,
)
from utils.funs_evaluate import create_run_dir
from utils.funs_evaluate import evaluate_sumo_along_axes
from nih_utils import (
    normalize_nih_results,
    nih_label_conversion,
    get_nih_inputs_outputs,
)


##################################################################
NORMALIZING_FUNCTION: Callable = normalize_nih_results
LABEL_CONVERSION_FUNCTION: Callable = nih_label_conversion
MAKE_LOG = True
if __name__ == "__main__":
    # load (and process) data (inc taking logs)
    TRAINING_FILE = Path("./data/results_Final_50LHS_TitrationProcessed.csv")
    run_dir = create_run_dir(Path("."), "evaluate")
    TRAINING_FILE = Path(shutil.copy(TRAINING_FILE, run_dir))
    input_vars, output_vars = get_nih_inputs_outputs(TRAINING_FILE)
    output_response = output_vars[-3]
    PROCESSED_TRAINING_FILE = process_input_file(
        TRAINING_FILE,
        make_log=MAKE_LOG,
        custom_operations=NORMALIZING_FUNCTION,
        columns_to_keep=input_vars + [output_response],
    )
    if MAKE_LOG:  # FIXME for now log applies to all inputs & the output
        input_vars = [f"log_{var}" for var in input_vars]
        output_response = f"log_{output_response}"

    evaluate_sumo_along_axes(
        run_dir,
        PROCESSED_TRAINING_FILE,
        input_vars,
        output_response,
        label_converter=LABEL_CONVERSION_FUNCTION,
        sumo_import_name="NIH_SuMo",
    )

    print("DONE")
