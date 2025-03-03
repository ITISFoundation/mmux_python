from pathlib import Path

import shutil
import sys

sys.path.append(str(Path(__file__).parent.parent))
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

if __name__ == "__main__":
    # load (and process) data (inc taking logs)
    TRAINING_FILE = Path("./data/results_Final_50LHS_TitrationProcessed.csv")
    run_dir = create_run_dir(Path("."), "evaluate")
    TRAINING_FILE = Path(shutil.copy(TRAINING_FILE, run_dir))
    var_names = get_variable_names(TRAINING_FILE)
    PROCESSED_TRAINING_FILE = process_input_file(
        TRAINING_FILE, make_log=True, custom_operations=normalize_nih_results
    )
    var_names = get_variable_names(PROCESSED_TRAINING_FILE)
    input_vars, output_vars = get_nih_inputs_outputs(PROCESSED_TRAINING_FILE)
    PROCESSED_TRAINING_FILE = process_input_file(
        PROCESSED_TRAINING_FILE, columns_to_remove=output_vars
    )

    evaluate_sumo_along_axes(
        run_dir,
        PROCESSED_TRAINING_FILE,
        input_vars,
        output_vars[-1],
        label_converter=nih_label_conversion,
    )

    print("DONE")
