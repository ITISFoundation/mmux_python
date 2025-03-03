import sys
from pathlib import Path
import shutil
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.funs_data_processing import process_input_file
from utils.funs_evaluate import create_run_dir, propagate_uq
from NIHinSilico.nih_utils import (
    normalize_nih_results,
    get_nih_inputs_outputs,
    nih_label_conversion,
)

NORMALIZING_FUNCTION: Callable = normalize_nih_results
LABEL_CONVERSION_FUNCTION: Callable = nih_label_conversion

## FOR NOW, JUST MAKE WORK, and GET PLOT w NormT for REPORT
make_log = True
if __name__ == "__main__":
    TRAINING_FILE = Path("./data/results_Final_50LHS_TitrationProcessed.csv")
    run_dir = create_run_dir(Path("."), "uq")
    TRAINING_FILE = Path(shutil.copy(TRAINING_FILE, run_dir))
    input_vars, output_vars = get_nih_inputs_outputs(TRAINING_FILE)
    ## TODO can select only thermal or only EM input vars?
    output_response = output_vars[-3]  ## FIXME just for now
    PROCESSED_TRAINING_FILE = process_input_file(
        TRAINING_FILE,
        make_log=make_log,
        custom_operations=NORMALIZING_FUNCTION,
        columns_to_keep=input_vars + [output_response],
    )

    means = {
        "SigmaMuscle": 0.46,
        "SigmaEpineurium": 0.0826,
        "SigmaPerineurium": 0.0021,
        "SigmaAlongFascicles": 0.571,
        "SigmaTransverseFascicles": 0.0826,
        "ThermalConductivity_Saline": 0.49,
        "ThermalConductivity_Fascicles": 0.48,
        "ThermalConductivity_Connective_Tissue": 0.39,
        "HeatTransferRate_Fascicles": 14896,
        "HeatTransferRate_Saline": 2722,
        "HeatTransferRate_Connective_Tissue": 2565,
    }
    stds = {
        "SigmaMuscle": 1.4,
        "SigmaEpineurium": 1.4,
        "SigmaPerineurium": 1.4,
        "SigmaAlongFascicles": 1.4,
        "SigmaTransverseFascicles": 1.4,
        "ThermalConductivity_Fascicles": 1.1,
        "ThermalConductivity_Saline": 1.08,
        "ThermalConductivity_Connective_Tissue": 1.128,
        "HeatTransferRate_Fascicles": 1.4,
        "HeatTransferRate_Saline": 1.35,
        "HeatTransferRate_Connective_Tissue": 1.4,
    }

    ## refactor as a function - it is pretty general, just need to pass in the means & stds
    if make_log:  # FIXME for now log applies to all inputs & the output
        input_vars = [f"log_{var}" for var in input_vars]
        output_response = f"log_{output_response}"
        means = {f"log_{key}": np.log(val) for key, val in means.items()}
        stds = {f"log_{key}": np.log(val) for key, val in stds.items()}

    savepath = propagate_uq(
        PROCESSED_TRAINING_FILE,
        run_dir,
        input_vars,
        output_response,
        means,
        stds,
        make_log,
        xscale="linear",
        yscale="linear",
        label_converter=LABEL_CONVERSION_FUNCTION,
    )
