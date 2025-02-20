import pandas as pd
import numpy as np
from pathlib import Path
import sys
import shutil
from typing import Callable

sys.path.append(str(Path(__file__).parent.parent.parent))
from tests.test_utils.test_funs_git import (
    create_run_dir,
    get_attr_from_repo,
)
from utils.funs_create_dakota_conf import (
    create_function_evaluation,
)
from utils_spinal import (
    get_model_from_spinal_repo,
    postpro_spinal_samples,
    create_sinusoid_pulse,
)
from utils.funs_data_processing import load_data
from utils.dakota_object import DakotaObject, Map
from utils.funs_plotting import plot_objective_space

## Default values
MAXAMP = 10.0
FREQPOS = 0
NFREQS = 20

AMP = 5.0
NUM_SAMPLES = 50
N_RUNNERS = 10
SWEEP_MODE = "FREQ"  # "AMP" or "FREQ"

run_dir = create_run_dir(Path.cwd(), "evaluation")
model = get_model_from_spinal_repo(run_dir)

shutil.copytree("GAF_kernels", run_dir / "GAF_kernels")
create_pulse: Callable = get_attr_from_repo(run_dir, "get_pulse.py", "get_pulse")
SEGMENT_PW = get_attr_from_repo(
    run_dir, module_name="get_pulse.py", function_name="SEGMENT_PW"
)
DURATION = get_attr_from_repo(
    run_dir, module_name="get_pulse.py", function_name="DURATION"
)
PULSE_FILE = "./SinusoidPulses.csv"


if SWEEP_MODE == "AMP":
    ## Option 1 - sweep amplitudes
    pulse_list = [
        create_sinusoid_pulse(amp=amp, freqpos=FREQPOS, NFREQS=NFREQS)[0]  # type: ignore
        for amp in np.linspace(0, MAXAMP, NUM_SAMPLES)
    ]
elif SWEEP_MODE == "FREQ":
    ## Option 2 - sweep pulse-widths
    freqpos_list = np.arange(NFREQS)
    pulse_list = [
        create_sinusoid_pulse(amp=AMP, freqpos=freqpos, NFREQS=NFREQS)[0]  # type: ignore
        for freqpos in freqpos_list
    ]
else:
    raise ValueError("Invalid SWEEP_MODE")

_, var_names = create_sinusoid_pulse(amp=AMP, freqpos=FREQPOS, NFREQS=NFREQS)
model.__annotations__["inputs"] = {var: float for var in var_names}

df = pd.DataFrame(pulse_list, columns=var_names)
df.to_csv(PULSE_FILE, index=False, sep=" ")
PULSE_FILE = Path(shutil.copy(PULSE_FILE, run_dir))
dakota_conf = create_function_evaluation(fun=model, samples_file=PULSE_FILE)

# run/retrieve Dakota sampling file
map = Map(model, n_runners=N_RUNNERS)
dakobj = DakotaObject(map)
dakobj.run(dakota_conf, output_dir=run_dir)


## analyze the results
evaluate_results_df = load_data(run_dir / "results.dat")
evaluate_results_df = postpro_spinal_samples(evaluate_results_df)

if SWEEP_MODE == "AMP":
    xvar = "Maximum Amplitude"
    xlim = (0, MAXAMP)
elif SWEEP_MODE == "FREQ":
    xvar = "Freq (kHz)"
    freq_list = [i / DURATION for i in freqpos_list]
    xlim = (np.min(freq_list), np.max(freq_list))
    evaluate_results_df[xvar] = freq_list

plot_objective_space(
    evaluate_results_df,
    xvar=xvar,
    yvar="Activation (%)",
    xlim=xlim,  # type: ignore
    ylim=(0, 100),
    title="Evaluated Objective Space",
    facecolors="none",
    savedir=run_dir,
    savefmt="png",
)

# plot_objective_space(
#     evaluate_results_df,
#     xvar=xvar,
#     yvar="FSI",
#     xlim=xlim,  # type: ignore
#     ylim=(0, 1),
#     title="Evaluated Objective Space",
#     facecolors="none",
#     savedir=run_dir,
#     savefmt="png",
# )

print("Done!")
