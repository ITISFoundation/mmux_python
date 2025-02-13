from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.funs_data_processing import load_data, get_non_dominated_indices
from utils.funs_plotting import plot_objective_space


from create_sampling_optistim import postpro_optistim_samples

run_dir = Path("runs/dakota_20250212.20090512_opt")

# analyze results (save plots; for git logs)
results_df = load_data(run_dir / "results.dat")
results_df = postpro_optistim_samples(results_df)
non_dominated_indices = get_non_dominated_indices(
    results_df,
    optimized_vars=["Energy", "Activation (%)"],
)
results_df["Activation (%)"] = -results_df["Activation (%)"]
plot_objective_space(
    results_df,
    non_dominated_indices=non_dominated_indices,
    xvar="Energy",
    yvar="Activation (%)",
    hvar="%eval_id",
    ylim=(0, 100),
    xlabel="Relative Energy (au)",
    ylabel="Activation (%)",
    title="Sampled Objective Space",
    facecolors="none",
    savedir=run_dir,
    savefmt="png",
)

print("Done!")
