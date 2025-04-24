from pathlib import Path
from typing import List, Dict, Callable, Optional
import numpy as np
import logging
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="flask_workflows.log",
    encoding="utf-8",
    level=logging.INFO,
    filemode="w",
)
logger.info("Logging started")


import shutil
import os
from utils.funs_data_processing import (
    get_variable_names,
    process_input_file,
)
from utils.funs_evaluate import create_run_dir
from utils.funs_evaluate import evaluate_sumo_along_axes, propagate_uq
from flask import Flask, request  # type: ignore

app = Flask(__name__)
base_dir = Path("/home/ordonez/mmux/mmux_vite/flaskapi")
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app) # allow CORS for all domains on all routes.
app.config['CORS_HEADERS'] = 'Content-Type'
#########################################################################
#########################################################################
## NIH-in-Silico specific logic -- ideally, we would eventually remove it, have agnostic
# (but useful) workflows, and some way to do / pass the normalization, relabeling...
from scripts.NIHinSilico.sumo_visualization_nih import (
    nih_label_conversion,
    normalize_nih_results,
)
from scripts.NIHinSilico.nih_utils import get_nih_inputs_outputs

NORMALIZING_FUNCTION: Callable = normalize_nih_results
LABEL_CONVERSION_FUNCTION: Callable = nih_label_conversion


## custom functionality while the FunctionsAPI is not yet available
# ideally we would register that function; and available datapoints as an existing JobCollection
@app.route("/flask/get_nih_inputs_outputs")
@cross_origin()
def flask_get_nih_inputs_outputs() -> Dict[str, List[str]]:
    logger.info("Starting flask function: flask_get_nih_inputs_outputs")
    logger.info("Cwd: " + str(Path.cwd()))
    filename = request.args.get("filename")
    logger.info("Inputs of the request: ", request.args)
    TRAINING_FILE = base_dir / "mmux_python" / "data" / filename
    logger.info(f"TRAINING_FILE: {TRAINING_FILE} does exist: {TRAINING_FILE.exists()}")
    var_names = get_variable_names(TRAINING_FILE)
    logger.info(f"var_names: {var_names}")

    ## NIH-in-Silico specific logic
    input_vars, output_vars = get_nih_inputs_outputs(TRAINING_FILE)
    logger.info(f"input_vars: {input_vars}")
    logger.info(f"output_vars: {output_vars}")
    return {"input_vars": input_vars, "output_vars": output_vars}

@app.route("/flask/retrieve_csv_result")
def retrieve_csv_result(
    csv_file_path: str, inputs: Dict[str, float], outputs: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Retrieve the result from a csv file.
    """

    df = pd.read_csv(csv_file_path)

    for col in inputs.keys():
        if col not in df.columns:
            raise ValueError(
                f"Input {col} not in the csv file. Columns are: {df.columns.values}"
            )
            
    if outputs is not None:
        for col in outputs:
            if col not in df.columns:
                raise ValueError(
                    f"Output {col} not in the csv file. Columns are: {df.columns.values}"
                )
        result = df.loc[np.all(df[inputs.keys()] == inputs.values(), axis=1), outputs]
    else:
        result = df.loc[np.all(df[inputs.keys()] == inputs.values(), axis=1)]
    # Check if the result is empty or has multiple rows
    assert len(result) != 0, f"No result found for inputs {inputs}."
    assert len(result) == 1, f"Multiple results found for inputs {inputs}."

    return result.iloc[0].to_dict()

#########################################################################
#########################################################################


def _save_in_react_public_folder(savepath: Path):
    # Copy the image to the react public folder
    logger.info(f"savepath: {savepath}")
    react_folder = Path("/home/ordonez/mmux/mmux_react/public/results")
    if react_folder.exists():
        logger.info(f"React folder exists at {react_folder}")
    else:
        logger.info(f"React folder does not exist at {react_folder}. Creating it.")
        react_folder.mkdir()
        logger.info(f"React folder created at {react_folder}: {react_folder.exists()}")

    # Check if the image already exists in the react public folder and remove it if it does
    output_image_path = react_folder / savepath.name
    if output_image_path.exists():
        logger.info(f"Existing image found at {output_image_path}. Removing it.")
        output_image_path.unlink()
        logger.info(f"Removed existing image at {output_image_path}")
    else:
        logger.info(f"No existing image found at {output_image_path}")
    savepath = shutil.copy(savepath, react_folder / savepath.name)
    logger.info(f"Image copied to {savepath}")
    logger.info(f"Returning {savepath.name}")
    logger.info("Done!")
    print("Done!")

    return {"imagePath": savepath.name}


@app.route("/flask/sumo_along_axes")
def flask_evaluate_sumo_along_axes() -> Dict[str, str]:
    os.chdir(Path(__file__).parent)
    logger.info("Starting flask function: flask_evaluate_sumo_along_axes")
    logger.info("Cwd: " + str(Path.cwd()))
    logger.info("Inputs of the request: ", request.args)
    training_file = request.args.get("filename")
    output_response = request.args.get("output")
    input_vars = request.args.get("inputs").split(",")
    make_log = False if request.args.get("log").lower() == "false" else True
    logger.info(f"training_file: {training_file}")
    logger.info(f"output_response: {output_response}")
    logger.info(f"input_vars: {input_vars}")
    logger.info(
        f"make_log: {make_log} (input {request.args.get('log')}) type: {type(make_log)}"
    )
    TRAINING_FILE = base_dir / "mmux_python" / "data" / training_file
    logger.info(f"TRAINING_FILE: {TRAINING_FILE} does exist: {TRAINING_FILE.exists()}")
    run_dir = create_run_dir(Path("."), "evaluate")
    TRAINING_FILE = Path(shutil.copy(TRAINING_FILE, run_dir))

    PROCESSED_TRAINING_FILE = process_input_file(
        TRAINING_FILE,
        make_log=make_log,
        columns_to_keep=input_vars + [output_response],
        custom_operations=NORMALIZING_FUNCTION,
    )
    if make_log:  # FIXME for now log applies to all inputs & the output
        input_vars = [f"log_{var}" for var in input_vars]
        output_response = f"log_{output_response}"

    savepath = evaluate_sumo_along_axes(
        run_dir,
        PROCESSED_TRAINING_FILE,
        input_vars,
        output_response,
        label_converter=LABEL_CONVERSION_FUNCTION,
    )
    # Copy the image to the react public folder
    _save_in_react_public_folder(savepath)
    return {"imagePath": savepath.name}


@app.route("/flask/uq_propagation")
def flask_uq_propagation() -> Dict[str, str]:
    os.chdir(Path(__file__).parent)
    logger.info("Starting flask function: flask_uq_propagation")
    logger.info("Cwd: " + str(Path.cwd()))
    logger.info("Inputs of the request: ", request.args)
    training_file = request.args.get("filename")
    output_response = request.args.get("output")
    input_vars = request.args.get("inputs").split(",")
    make_log = False if request.args.get("log").lower() == "false" else True
    logger.info(f"training_file: {training_file}")
    logger.info(f"output_response: {output_response}")
    logger.info(f"input_vars: {input_vars}")
    logger.info(
        f"make_log: {make_log} (input {request.args.get('log')}) type: {type(make_log)}"
    )
    TRAINING_FILE = base_dir / "mmux_python" / "data" / training_file
    logger.info(f"TRAINING_FILE: {TRAINING_FILE} does exist: {TRAINING_FILE.exists()}")
    run_dir = create_run_dir(Path("."), "uq")
    TRAINING_FILE = Path(shutil.copy(TRAINING_FILE, run_dir))

    PROCESSED_TRAINING_FILE = process_input_file(
        TRAINING_FILE,
        make_log=make_log,
        columns_to_keep=input_vars + [output_response],
        custom_operations=NORMALIZING_FUNCTION,
    )

    # TODO make available as JSON from the training data to React? To display as defaults?
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
        xscale="linear",
        label_converter=LABEL_CONVERSION_FUNCTION,
    )
    _save_in_react_public_folder(savepath)
    return {"imagePath": savepath.name}
