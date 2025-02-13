from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from tests.test_utils.test_funs_git import get_model_from_optistim_repo, create_run_dir
from utils.funs_create_dakota_conf import create_optimization_moga
from utils.funs_data_processing import load_data, get_non_dominated_indices
from utils.dakota_object import DakotaObject, Map
from utils.funs_plotting import plot_objective_space

MAXAMP = 7.50
NUM_SAMPLES = 5000
N_RUNNERS = 16

########################################################################################
if __name__ == "__main__":
    # clone optistim repo
    run_dir = create_run_dir(Path.cwd(), "opt")
    model = get_model_from_optistim_repo(run_dir)
    from typing import Callable, Literal, List

    ## TODO could I even integrate it in the Map object??
    class MinimizationModel:
        def __init__(
            self,
            model: Callable,
            optimization_modes: List[Literal["max", "min"]],
        ):
            outputs = model.__annotations__["outputs"]
            assert len(optimization_modes) == len(outputs)

            self.model = model
            self.optimization_modes = optimization_modes
            ## FIXME I am chaning the label in the response, but not in the signature,
            # and thus not in the dakota file. Will that give me errors?

        def run(self, *args, **kwargs):
            results = self.model(*args, **kwargs)
            if isinstance(results, (int, float)):
                results = [results]
            if not isinstance(results, dict):
                results = {f"fn{i+1}": val for i, val in enumerate(results)}
            ## from here on, results SHOULD BE a dictionary
            assert len(results) == len(self.optimization_modes)

            for m, k in zip(self.optimization_modes, results.keys()):
                if m == "min":
                    continue
                elif m == "max":
                    results.update({k: -results[k]})
                else:
                    raise ValueError("Only min or max are allowed")
            return results

            # results_for_minimization = {}
            # for mode, result, key in zip(
            #     self.optimization_modes, results.values(), results.keys()
            # ):
            #     if mode == "min":
            #         results_for_minimization.update({key: result})
            #     elif model == "max":
            #         results_for_minimization.update({key: -1 * result})
            #         ## no changing of keys, just complicates things
            #         ## simply making it negative -- that could be easily changed in the plot,
            #         ## if at all necessary

            # return results_for_minimization

    # create Dakota sampling file
    dakota_conf = create_optimization_moga(
        fun=model,
        moga_kwargs={"max_function_evaluations": NUM_SAMPLES},
        batch_mode=True,
        lower_bounds=[-MAXAMP for _ in range(len(model.__annotations__["inputs"]))],
        upper_bounds=[MAXAMP for _ in range(len(model.__annotations__["inputs"]))],
        # optimization_modes=["max", "min", "min"],
    )

    # run/retrieve Dakota sampling file
    map = Map(
        model=MinimizationModel(model, ["max", "min", "min"]).run,
        n_runners=N_RUNNERS,
    )
    dakobj = DakotaObject(map)

    import matplotlib.pyplot as plt
    from typing import List, Optional
    import pandas as pd
    from pathlib import Path
    from matplotlib.axes import Axes

    def plot_optimization_evolution(
        df: pd.DataFrame,
        output_vars: List[str],
        savepath: Optional[Path] = None,
        savedir: Optional[Path] = None,
        axs: Optional[List[Axes]] = None,
    ):
        for var in output_vars:
            assert var in df, f"Output variable {var} not found in DataFrame"

        if axs is None:
            fig, axs = plt.subplots(
                1, len(output_vars), figsize=(len(output_vars) * 4, 4), sharex=True
            )
            axs = axs.flatten()  # type:ignore
        assert axs is not None

        window = 100  # Define the window size for the moving average
        for i, var in enumerate(output_vars):
            ax = axs[i]
            rolling_mean = df[var].rolling(window=window).mean()
            rolling_min = df[var].rolling(window=window).min()
            rolling_max = df[var].rolling(window=window).max()

            ax.plot(rolling_mean, label=f"{var} (mean)")
            ax.fill_between(
                range(len(df[var])),
                rolling_min,
                rolling_max,
                alpha=0.2,
                label=f"{var} (min-max)",
            )
            ax.set_xlabel("Evaluation")
            ax.set_title(var)

        plt.suptitle("Current optimization evolution", fontsize=14)
        plt.tight_layout()
        if savepath is None:
            if savedir is None:
                savedir = Path(".")
            savepath = savedir / "optimization_evolution.png"
        plt.savefig(savepath, format="png", dpi=300)
        print(f"Figure saved in {savepath}")
        return savepath

    # dakobj.run(dakota_conf, output_dir=run_dir)
    import threading
    import time

    REFRESH_RATE = 5.0  ## IN SECONDS
    savepath = run_dir / "optimization_evolution.png"
    savepath.touch()
    t = threading.Thread(target=dakobj.run, args=(dakota_conf, run_dir))
    t.start()
    while t.is_alive():  # this seems to time it rather ok :)
        if (run_dir / "results.dat").is_file() and (
            (time.time() - savepath.stat().st_mtime) >= REFRESH_RATE
        ):
            print("Generating plot with current Dakota optimziation results...")
            results_df = load_data(run_dir / "results.dat")
            plot_optimization_evolution(
                results_df,
                model.__annotations__["outputs"],
                savepath=savepath,
            )
    t.join()

    from create_sampling_optistim import postpro_optistim_samples

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
