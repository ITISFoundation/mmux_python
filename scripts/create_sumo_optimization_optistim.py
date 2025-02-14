from pathlib import Path
import sys
from typing import Callable, Literal, List
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from tests.test_utils.test_funs_git import (
    get_model_from_optistim_repo,
    create_run_dir,
    get_attr_from_repo,
)
from utils.funs_create_dakota_conf import create_optimization_moga
from utils.funs_data_processing import load_data, get_non_dominated_indices
from utils.dakota_object import DakotaObject, Map
from utils.funs_plotting import plot_objective_space, plot_optimization_evolution

MAXAMP = 7.50
N_RUNNERS = 8

########################################################################################
if __name__ == "__main__":
    ## avoid circular import if importing at the main class
    from create_sampling_optistim import postpro_optistim_samples

    # clone optistim repo
    run_dir = create_run_dir(Path.cwd(), "opt")
    model = get_model_from_optistim_repo(run_dir)

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
        moga_kwargs={
            "max_function_evaluations": 1e4,
            ## hyperparams below created well distributed front in SCS example. Let's try.
            "population_size": 100,
            "max_iterations": 1000,
            "radial_distances": [0.1, 0.1, 0.1],
            "seed": 42,
        },
        batch_mode=True,
        lower_bounds=[-MAXAMP for _ in range(len(model.__annotations__["inputs"]))],
        upper_bounds=[MAXAMP for _ in range(len(model.__annotations__["inputs"]))],
    )

    # run/retrieve Dakota sampling file
    map = Map(
        model=MinimizationModel(model, ["max", "min", "min"]).run,
        n_runners=N_RUNNERS,
    )
    dakobj = DakotaObject(map)

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
            results_df = postpro_optistim_samples(results_df)
            results_df["Activation (%)"] = -results_df["Activation (%)"]
            plot_optimization_evolution(
                results_df,
                ["Activation (%)", "Energy", "Maximum Amplitude"],
                savepath=savepath,
            )
            plot_objective_space(
                results_df,
                non_dominated_indices=list(results_df[-100:].index.values),
                xvar="Energy",
                yvar="Activation (%)",
                # hvar="%eval_id",
                ylim=(0, 100),
                title="Running Optimization - Objective Space",
                facecolors="none",
                savedir=run_dir,
                savefmt="png",
            )
    t.join()

    # analyze results (save plots; for git logs)
    results_df = load_data(run_dir / "results.dat")
    results_df = postpro_optistim_samples(results_df)
    non_dominated_indices = get_non_dominated_indices(
        results_df,
        optimized_vars=["Energy", "Activation (%)"],
        sort_by_column="Energy",
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

    create_pulse: Callable = get_attr_from_repo(run_dir, "get_pulse.py", "get_pulse")

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

    print("Done!")
