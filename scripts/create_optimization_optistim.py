from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from tests.test_utils.test_funs_git import get_model_from_optistim_repo, create_run_dir
from utils.funs_create_dakota_conf import create_optimization_moga
from utils.funs_data_processing import load_data
from utils.dakota_object import DakotaObject, Map

MAXAMP = 7.50
NUM_SAMPLES = 1000
N_RUNNERS = 32

########################################################################################
if __name__ == "__main__":
    # clone optistim repo
    run_dir = create_run_dir(Path.cwd(), "opt")
    model = get_model_from_optistim_repo(run_dir)

    # create Dakota sampling file
    dakota_conf = create_optimization_moga(
        fun=model,
        moga_kwargs={"max_function_evaluations": NUM_SAMPLES},
        batch_mode=True,
        lower_bounds=[-MAXAMP for _ in range(len(model.__annotations__["inputs"]))],
        upper_bounds=[MAXAMP for _ in range(len(model.__annotations__["inputs"]))],
    )

    # run/retrieve Dakota sampling file
    map = Map(model, n_runners=N_RUNNERS)
    dakobj = DakotaObject(map)

    import matplotlib.pyplot as plt
    from typing import List, Optional
    import pandas as pd
    from pathlib import Path
    from matplotlib.axes import Axes

    def plot_optimization_evolution(
        df: pd.DataFrame,
        output_vars: List[str],
        savedir: Path,
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

        for i, var in enumerate(output_vars):
            ax = axs[i]
            ax.plot(df[var], label=var)
            ax.set_xlabel("Evaluation")
            ax.set_title(var)

        plt.suptitle("Optimization evolution", fontsize=20)
        savepath = savedir / "optimization_evolution.png"
        plt.savefig(savepath, format="png", dpi=300)
        print(f"Figure saved in {savepath}")
        return

    # dakobj.run(dakota_conf, output_dir=run_dir)
    import threading

    t = threading.Thread(target=dakobj.run, args=(dakota_conf, run_dir))
    t.start()
    while t.is_alive():  # this seems to time it rather ok :)
        # time.sleep(1)  ## for some reason this causes the EOFError
        print("Waiting for Dakota to finish")
        optimization_results_df = load_data(run_dir / "results.dat")
        plot_optimization_evolution(
            optimization_results_df, model.__annotations__["outputs"], run_dir
        )
    t.join()
    # analyze results (save plots; for git logs)
    sampling_results_df = load_data(run_dir / "results.dat")

    ## TODO analyze results (real time, in a separate script??)
