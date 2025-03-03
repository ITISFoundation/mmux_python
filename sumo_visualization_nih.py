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
N_INPUTS = 5 + 6  # EM + Thermal
N_OUTPUTS = (
    4  #### Current Shunting
    + 4 * 3  #### Dosimetry in 3 areas
    + 3  #### Thermal
    # + 7  #### Neuro
)
##################################################################


def normalize_nih_results(df: pd.DataFrame):
    """Custom operations for NIH dataset.
    Normalize Dosimetric amounts (current Shunting; EM Dosimetry) by current (wasnt done in simulation analysis).
    Normalize Thermal peaks by total current. This normalization is already in the simulation,
    but Thermal deposition scales with power (e.g. square of current) so an additional normalization is necessary.
    """
    for c in df.columns:
        assert (
            "log" not in c
        ), "Log found. The normalization operation assumes variables in linear space."

        if ("Thermal_Peak" in c) or ("Dosimetry" in c) or ("Shunting" in c):
            if not (c == "EM_Shunting_Total_Current"):
                df[c] = df[c] / df["EM_Shunting_Total_Current"]

    return df


def nih_label_conversion(old_label: str) -> str:
    assert isinstance(
        old_label, str
    ), f"old_label must be a string object! Instead, {type(old_label)} found."

    ## pattern exceptions
    if "EM_Shunting_Shunted_Current" in old_label:
        return "Shunted Current Outside the Nerve (%)"

    # tissues
    if "Saline" in old_label or "Muscle" in old_label or "Outside" in old_label:
        tissue = "Muscle "
    elif "Nerve" in old_label:
        tissue = "Nerve "
    elif "Fascicle" in old_label:
        tissue = "Fascicle "
        if "Transverse" in old_label:
            tissue += "Transversal "
        elif "Along" in old_label:
            tissue += "Longitudinal "
    elif "Total" in old_label or "Overall" in old_label:
        tissue = "Total "
    elif "Epineurium" in old_label:
        tissue = "Epineurium "
    elif "Perineurium" in old_label:
        tissue = "Perineurium "
    elif "Connective_Tissue" in old_label:
        tissue = "Nerve (Connective Tissue) "
    else:
        raise ValueError("Tissue not matched for " + old_label)

    #

    if "Shunting" in old_label and "Current" in old_label:
        return tissue + "Shunted Current"
    elif "PeakE" in old_label:
        return tissue + "Peak E-field"
    elif "Iso99E" in old_label:
        return tissue + "99% Iso-Percentile E-field"
    elif "Iso98E" in old_label:
        return tissue + "98% Iso-Percentile E-field"
    elif "icnirp_peaks" in old_label:
        return tissue + "ICNIRP Peak"
    elif "Thermal_Peak" in old_label:
        return tissue + "Peak Temperature Increase"
    elif "EM_Neuro" in old_label:
        if "Threshold" in old_label:
            return " ".join(old_label.split("_")[-3:])
        elif "Quantile" in old_label:
            q = old_label.split("_")[-1].split("percent")[0]
            return f"Neuro {q}% Quantile"
        else:
            raise ValueError("Did not find right keywork in Neuro label")
    elif "ThermalConductivity" in old_label:
        return tissue + "\nThermal Conductivity ($\\kappa$)"
    elif "HeatTransferRate" in old_label:
        return (
            tissue + "\nBlood Perfusion ($\\omega$)"
        )  ## TODO double check in report - and add symbol? :)
    elif "Sigma":
        return tissue + "\nElectric Conductivity ($\\sigma$)"
    else:
        raise ValueError("Did not find which label to assign")


##################################################################

if __name__ == "__main__":
    # load (and process) data (inc taking logs)
    TRAINING_FILE = Path("./data/results_Final_50LHS_TitrationProcessed.csv")
    run_dir = create_run_dir(Path("."), "evaluate")
    TRAINING_FILE = Path(shutil.copy(TRAINING_FILE, run_dir))
    var_names = get_variable_names(TRAINING_FILE)
    assert len(var_names) == N_INPUTS + N_OUTPUTS, (
        f"Number of variables in data {len(var_names)} does not coincide with "
        f"expected number of input variables {N_INPUTS} and output responses {N_OUTPUTS}"
    )
    PROCESSED_TRAINING_FILE = process_input_file(
        TRAINING_FILE, make_log=True, custom_operations=normalize_nih_results
    )
    var_names = get_variable_names(PROCESSED_TRAINING_FILE)
    input_vars = var_names[:N_INPUTS]
    output_vars = var_names[-N_OUTPUTS:]
    ### FIXME not able to give only some output vars; would need to
    # remove them from the (PROCESSED_)TRAINING_FILE as well
    PROCESSED_TRAINING_FILE = process_input_file(
        PROCESSED_TRAINING_FILE, columns_to_remove=output_vars[:12]
    )

    evaluate_sumo_along_axes(
        run_dir,
        PROCESSED_TRAINING_FILE,
        input_vars,
        output_vars[12:],
        label_converter=nih_label_conversion,
    )

    print("DONE")
