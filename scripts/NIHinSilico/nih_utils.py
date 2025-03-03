import pandas as pd
from pathlib import Path
from typing import List, Tuple
from utils.funs_data_processing import get_variable_names
import logging

logger = logging.getLogger(__name__)


def get_nih_inputs_outputs(TRAINING_FILE: Path) -> Tuple[List[str], List[str]]:
    ## NIH-in-Silico specific logic
    var_names = get_variable_names(TRAINING_FILE)
    input_vars, output_vars = [], []
    for var in var_names:
        if (
            ("Sigma" in var)
            or ("ThermalConductivity" in var)
            or ("HeatTransferRate" in var)
        ):
            input_vars.append(var)
            # input_vars.append(nih_label_conversion(var))
        elif ("Thermal_Peak" in var) or ("Dosimetry" in var) or ("EM_Shunting" in var):
            output_vars.append(var)
            # output_vars.append(nih_label_conversion(var))
        else:
            logger.warning(
                f"Variable {var} not recognized as input or output. Ignoring it."
            )
            raise ValueError(f"Variable {var} not recognized as input or output.")
    logger.info(f"input_vars: {input_vars}")
    logger.info(f"output_vars: {output_vars}")
    return input_vars, output_vars


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
