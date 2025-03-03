import sys
from pathlib import Path
import shutil
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.dakota_object import DakotaObject
from utils.funs_create_dakota_conf import create_uq_propagation
from utils.funs_data_processing import (
    process_input_file,
    get_results,
)
from utils.funs_evaluate import create_run_dir
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
        custom_operations=normalize_nih_results,
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

    ## TODO extract results from "predictions.dat" file and plot histogram (just one variable)
    fig, ax = plt.subplots()
    x = get_results(run_dir / f"predictions.dat", output_response)
    if make_log:
        x = np.exp(x)
    ax.hist(x, bins=50, density=False)
    ax.set_xlabel(LABEL_CONVERSION_FUNCTION(output_response))
    savefmt: str = "png"
    savepath = run_dir / (output_response + "." + savefmt)
    plt.savefig(savepath, format=savefmt, dpi=300)
    print(f"Figure saved in {savepath}")
    assert savepath is not None
    assert savepath.exists(), f"Plotting failed, savepath {savepath} does not exist"
