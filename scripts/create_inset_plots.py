from pathlib import Path
import sys
from typing import Callable
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))
from tests.test_utils.test_funs_git import (
    get_attr_from_repo,
)
from utils.funs_data_processing import load_data, get_non_dominated_indices


from create_sampling_optistim import postpro_optistim_samples
import numpy as np

run_dir = Path("runs/dakota_20250212.20090512_opt")
create_pulse: Callable = get_attr_from_repo(run_dir, "get_pulse.py", "get_pulse")
model: Callable = get_attr_from_repo(
    run_dir, module_name="evaluation.py", function_name="model"
)
results_df = load_data(run_dir / "results.dat")
results_df = postpro_optistim_samples(results_df)
non_dominated_indices = get_non_dominated_indices(
    results_df,
    optimized_vars=["Energy", "Activation (%)"],
    sort_by_column="Energy",
)
results_df["Activation (%)"] = -results_df["Activation (%)"]

# for i, idx in enumerate(non_dominated_indices):
#     r = results_df.loc[idx]
#     params = r[list(model.__annotations__["inputs"].keys())].values
#     pulse = create_pulse(*params)
#     pulse.std_list = None  ## need to fix that again. It is adding three zero values
#     # (was fixed in some other branch)
#     pulse.plot_pulse()
#     plt.title(
#         f"Activation = {r['Activation (%)']:.2f}%, Energy = {r['Energy']*1e6:.2f}uJ"
#     )
#     plt.savefig(run_dir / f"nondompulse{i}_sample{idx}.png", dpi=300)

NINTERVAL = 10
interval_pulses = []
for i in range(0, len(non_dominated_indices), NINTERVAL):
    idx_group = non_dominated_indices[i : i + NINTERVAL]
    group_pulses = []
    for idx in idx_group:
        r = results_df.loc[idx]
        params = r[list(model.__annotations__["inputs"].keys())].values
        pulse = create_pulse(*params)
        group_pulses.append(pulse.amplitude_list)
    if group_pulses:
        average_pulse = np.mean(group_pulses, axis=0)
        pulse.amplitude_list = np.mean(group_pulses, axis=0)
        pulse.std_list = np.std(group_pulses, axis=0)
        pulse.plot_pulse()
        plt.savefig(run_dir / f"nondompulse{i}-{i+NINTERVAL}_meanstd.png", dpi=300)
