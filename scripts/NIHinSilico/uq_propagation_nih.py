import sys
from pathlib import Path
import shutil

sys.path.append(str(Path(__file__).parent.parent))
from utils.funs_data_processing import (
    get_variable_names,
    process_input_file,
)
from utils.funs_evaluate import create_run_dir
from NIHinSilico.nih_utils import normalize_nih_results

if __name__ == "__main__":
    # load (and process) data (inc taking logs)
    TRAINING_FILE = Path("./data/results_Final_50LHS_TitrationProcessed.csv")
    run_dir = create_run_dir(Path("."), "uq")
    TRAINING_FILE = Path(shutil.copy(TRAINING_FILE, run_dir))
    var_names = get_variable_names(TRAINING_FILE)
    PROCESSED_TRAINING_FILE = process_input_file(
        TRAINING_FILE, make_log=True, custom_operations=normalize_nih_results
    )
