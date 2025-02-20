from pathlib import Path
import sys
import shutil

sys.path.append(str(Path(__file__).parent.parent.parent))
from tests.test_utils.test_funs_git import create_run_dir, get_attr_from_repo
from utils.funs_create_dakota_conf import create_function_sampling
from utils.funs_data_processing import load_data, get_non_dominated_indices
from utils.dakota_object import DakotaObject, Map
from utils.funs_plotting import plot_objective_space
from utils_spinal import (
    get_model_from_spinal_repo,
    postpro_spinal_samples,
    create_sinusoid_pulse,
)

MAXAMP = 7.50
NUM_SAMPLES = 1500
N_RUNNERS = 20

NFREQS = 10

########################################################################################
########################################################################################
if __name__ == "__main__":
    # clone optistim repo
    run_dir = create_run_dir(Path.cwd(), "sampling")
    model = get_model_from_spinal_repo(run_dir)
    shutil.copytree("GAF_kernels", run_dir / "GAF_kernels")
    _, var_names = create_sinusoid_pulse(NFREQS=NFREQS)
    model.__annotations__["inputs"] = {var: float for var in var_names}

    # create Dakota sampling file
    dakota_conf = create_function_sampling(
        fun=model,
        num_samples=NUM_SAMPLES,
        batch_mode=True,
        lower_bounds=[0.0 for _ in range(len(model.__annotations__["inputs"]))],
        upper_bounds=[
            MAXAMP / NFREQS for _ in range(len(model.__annotations__["inputs"]))
        ],
    )

    # run/retrieve Dakota sampling file
    map = Map(model, n_runners=N_RUNNERS)
    dakobj = DakotaObject(map)
    dakobj.run(dakota_conf, output_dir=run_dir)

    # analyze results (save plots; for git logs)
    sampling_results_df = load_data(run_dir / "results.dat")
    sampling_results_df = postpro_spinal_samples(sampling_results_df)
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
        vars=["Activation (%)", "FSI", "Energy", "Maximum Amplitude"],
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
