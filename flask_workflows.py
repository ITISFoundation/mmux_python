from pathlib import Path
import pandas as pd
from typing import List, Tuple
import shutil
import	os
from utils.funs_data_processing import (
    get_variable_names,
    process_input_file,
)
from typing import Dict

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename="sumo_visualization.log", encoding="utf-8", level=logging.INFO, filemode='w')
logger.info("Logging started")

from utils.funs_evaluate import create_run_dir
from utils.funs_evaluate import evaluate_sumo_along_axes
from sumo_visualization_nih import nih_label_conversion, normalize_nih_results

from flask import Flask, request
app = Flask(__name__)
base_dir = Path("/home/ordonez/mmux/mmux_react/api")

import time
@app.route('/api/time')
def get_current_time():
    print('get_current_time called')
    return {'time': str(time.time())}  # jsonified gets automatically called by Flask

## custom functionality while the FunctionsAPI is not yet available
# ideally we would register that function; and available datapoints as an existing JobCollection
@app.route('/flask/get_nih_inputs_outputs')
def flask_get_nih_inputs_outputs() -> Tuple[List[str], List[str]]:
    logger.info("Starting flask function: flask_get_nih_inputs_outputs")
    logger.info("Cwd: " + str(Path.cwd()))
    filename = request.args.get("filename")
    logger.info("Inputs of the request: ", request.args)
    TRAINING_FILE = base_dir / "mmux_python" / "data" / filename
    logger.info(f"TRAINING_FILE: {TRAINING_FILE} does exist: {TRAINING_FILE.exists()}")
    var_names = get_variable_names(TRAINING_FILE)
    logger.info(f"var_names: {var_names}")
    
    ## NIH-in-Silico specific logic
    input_vars, output_vars = [], []
    for var in var_names:
        if ("Sigma" in var) or ("ThermalConductivity" in var) or ("HeatTransferRate" in var): 
            input_vars.append(var)
            # input_vars.append(nih_label_conversion(var))
        elif ("Thermal_Peak" in var) or ("Dosimetry" in var) or ("EM_Shunting" in var):
            output_vars.append(var)
            # output_vars.append(nih_label_conversion(var))
        else:
            logger.warning(f"Variable {var} not recognized as input or output. Ignoring it.")
            raise ValueError(f"Variable {var} not recognized as input or output.")
    logger.info(f"input_vars: {input_vars}")
    logger.info(f"output_vars: {output_vars}")
    return {"input_vars": input_vars, "output_vars": output_vars}

@app.route('/flask/sumo_along_axes')
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
    logger.info(f"make_log: {make_log} (input {request.args.get('log')}) type: {type(make_log)}")
    TRAINING_FILE = base_dir / "mmux_python" / "data" / training_file
    logger.info(f"TRAINING_FILE: {TRAINING_FILE} does exist: {TRAINING_FILE.exists()}")
    run_dir = create_run_dir(Path("."), "evaluate")
    TRAINING_FILE = Path(shutil.copy(TRAINING_FILE, run_dir))
    
    PROCESSED_TRAINING_FILE = process_input_file(
        TRAINING_FILE, make_log=make_log, columns_to_keep=input_vars + [output_response],
        custom_operations=normalize_nih_results,
    )
    if make_log: # temporary; for now log applies to all inputs & the output
        input_vars = [f"log_{var}" for var in input_vars]
        output_response = f"log_{output_response}"
    
    savepath = evaluate_sumo_along_axes(
        run_dir,
        PROCESSED_TRAINING_FILE,
        input_vars,
        output_response,
        label_converter=nih_label_conversion,
    )
    
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

    return {'imagePath': savepath.name}