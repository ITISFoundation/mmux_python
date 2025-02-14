import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from tests.test_utils.test_funs_git import get_model_from_optistim_repo, create_run_dir
from utils.funs_create_dakota_conf import create_function_sampling
from utils.funs_data_processing import load_data, get_non_dominated_indices
from utils.dakota_object import DakotaObject, Map
from utils.funs_plotting import plot_objective_space

MAXAMP = 7.50
NUM_SAMPLES = 200
N_RUNNERS = 10


## e.g. previously called "load_data_dakota_free_pulse_optimization"
def postpro_optistim_samples(df: pd.DataFrame) -> pd.DataFrame:
    ## Add objective functions with proper names; leave the old ones as columns as well
    if "1-activation" in df.columns:
        df["Activation (%)"] = 100 * (1.0 - df["1-activation"])
        df.pop("1-activation")
    elif "activation" in df.columns:
        df["Activation (%)"] = 100 * df["activation"]
        df.pop("activation")

    if "energy" in df.columns:
        df["Energy"] = df["energy"]
        df.pop("energy")

    if "maxampout" in df.columns:
        df["Maximum Amplitude"] = df["maxampout"]
        df.pop("maxampout")
    elif "maxamp" in df.columns:
        df["Maximum Amplitude"] = df["maxamp"]
        df.pop("maxamp")

    return df


########################################################################################
########################################################################################
if __name__ == "__main__":
    # clone optistim repo
    run_dir = create_run_dir(Path.cwd(), "sampling")
    model = get_model_from_optistim_repo(run_dir)

    # create Dakota sampling file
    dakota_conf = create_function_sampling(
        fun=model,
        num_samples=NUM_SAMPLES,
        batch_mode=True,
        lower_bounds=[-MAXAMP for _ in range(len(model.__annotations__["inputs"]))],
        upper_bounds=[MAXAMP for _ in range(len(model.__annotations__["inputs"]))],
    )

    # run/retrieve Dakota sampling file
    map = Map(model, n_runners=N_RUNNERS)
    dakobj = DakotaObject(map)
    dakobj.run(dakota_conf, output_dir=run_dir)

    # analyze results (save plots; for git logs)
    sampling_results_df = load_data(run_dir / "results.dat")
    sampling_results_df = postpro_optistim_samples(sampling_results_df)
    non_dominated_indices = get_non_dominated_indices(
        sampling_results_df,
        optimized_vars=["Energy", "Activation (%)"],
        optimization_modes=["min", "max"],
    )
    plot_objective_space(
        sampling_results_df,
        non_dominated_indices=non_dominated_indices,
        xvar="Energy",
        yvar="Activation (%)",
        ylim=(0, 100),
        xlabel="Relative Energy (au)",
        ylabel="Activation (%)",
        title="Sampled Objective Space",
        facecolors="none",
        savedir=run_dir,
        savefmt="png",
    )
    ## TODO try to use an input schema form, or simply parameters here,
    # so it is clear for which input parameters we get which output plots
    # (save them to disk & git-track them)
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.pairplot(
        sampling_results_df,
        vars=["Activation (%)", "Energy", "Maximum Amplitude"],
        # kind="hist",
        diag_kind="kde",
        plot_kws={"alpha": 0.5},
    )
    plt.savefig(run_dir / "pairplot_outputs.png", format="png", dpi=300)

    NVARS = len(model.__annotations__["inputs"])
    sns.pairplot(
        sampling_results_df,
        x_vars=sampling_results_df.columns[2 : 2 + NVARS],
        y_vars=sampling_results_df.columns[2 + NVARS :],
        plot_kws={"alpha": 0.5},
    )
    plt.savefig(run_dir / "pairplot_inputs_outputs.png", format="png", dpi=300)
    print("DONE")
