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
from utils_spinal import get_model_from_spinal_repo, postpro_spinal_samples
from utils.funs_data_processing import load_data
from utils.dakota_object import DakotaObject, Map
from utils.funs_plotting import plot_objective_space

## Default values
PW = 0.3
MAXAMP = 5.0

AMP = 3.0
NUM_SAMPLES = 50
N_RUNNERS = 10
SWEEP_MODE = "PW"  # "AMP" or "PW"

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
NVARS = get_attr_from_repo(run_dir, module_name="get_pulse.py", function_name="NVARS")
PULSE_FILE = "./BiphasicPulses.csv"


def create_biphasic_pulse(pw=0.3, amp=1.0, MAKEPLOT=False):
    var_names = [f"p{i+1}" for i in range(NVARS)]
    nseg_per_pw = round(pw / SEGMENT_PW)
    assert (
        2 * nseg_per_pw < NVARS
    ), f"pulse-width of each phase ({pw}) must be less than half the total pulse duration {DURATION / 2}"

    vars = []
    for n in range(nseg_per_pw):
        vars.append(amp)
    for n in range(nseg_per_pw):
        vars.append(-amp)
    for n in range(NVARS - 2 * nseg_per_pw):
        vars.append(0.0)

    if MAKEPLOT:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 3))
        x = (np.arange(NVARS) + 0.5) * SEGMENT_PW
        plt.bar(x=x, height=vars, width=SEGMENT_PW)
        plt.hlines(0.0, 0, DURATION, "grey", "dashed", alpha=0.5)
        plt.xlabel("Time (ms)")
        plt.ylabel("Pulse amplitude")
        plt.show(block=False)

    return vars, var_names


if SWEEP_MODE == "AMP":
    ## Option 1 - sweep amplitudes
    pulse_list = [
        create_biphasic_pulse(pw=PW, amp=amp, MAKEPLOT=False)[0]  # type: ignore
        for amp in np.linspace(0, MAXAMP, NUM_SAMPLES)
    ]
elif SWEEP_MODE == "PW":
    ## Option 2 - sweep pulse-widths
    pw_list = np.linspace(0.0, 0.5 * (NUM_SAMPLES - 1) / NUM_SAMPLES, NUM_SAMPLES)
    pulse_list = [
        create_biphasic_pulse(pw=pw, amp=AMP, MAKEPLOT=False)[0]  # type: ignore
        for pw in pw_list
    ]
else:
    raise ValueError("Invalid SWEEP_MODE")

_, var_names = create_biphasic_pulse(pw=PW, amp=MAXAMP, MAKEPLOT=False)
df = pd.DataFrame(pulse_list, columns=var_names)
df.to_csv(PULSE_FILE, index=False, sep=" ")
PULSE_FILE = Path(shutil.copy(PULSE_FILE, run_dir))
dakota_conf = create_function_evaluation(fun=model, samples_file=PULSE_FILE)
# run/retrieve Dakota sampling file
map = Map(model, n_runners=N_RUNNERS)
dakobj = DakotaObject(map)
dakobj.run(dakota_conf, output_dir=run_dir)

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
elif SWEEP_MODE == "PW":
    xvar = "Pulse Width (ms)"
    xlim = (np.min(pw_list), np.max(pw_list))
    evaluate_results_df[xvar] = pw_list

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

print("Done!")
