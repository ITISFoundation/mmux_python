import pandas as pd
from typing import Tuple, Callable
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from tests.test_utils.test_funs_git import get_model_from_optistim_repo, create_run_dir
from utils.funs_create_dakota_conf import create_function_sampling
from utils.funs_data_processing import load_data, get_non_dominated_indices
from utils.dakota_object import DakotaObject, Map
from utils.funs_plotting import plot_objective_space

MAXAMP = 5.0

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

    return df


## TODO double check, maybe move somewhere else...
## Do I want to have a folder with diff parts (sampling, ...)
## and share the sampling file through a file indicating the path?
def add_inset_pulses(
    pulses_args: tuple,
    errors: tuple,
    pareto_point: Tuple[float, float],
    ax: Axes,
    create_pulses_fun: Callable,
    flip_x_y: bool = True,
    total_size: Tuple[float, float] = (0.5, 0.3),
    partial_size: float = 0.4 * 100,
    ylim: Tuple[float, float] = (-1e3, 1e3),
    x_offset: float = 0.0,
    y_offset: float = 0.0,
):
    """Generate an inset plot with the pulses used to generate the pareto point, within the pareto front plot itself.
    NB: a box is generated around the pareto point (using total_size), and then only the lower right corner of it is used (using partial_size).

    Args:
        pulses_args (tuple): Input used to generate the pulses (ie X[i])
        pareto_point (Tuple[float, float]): Ouput of the evaluation (ie F[i])
        ax (plt.Axes): Main axes in which to create the inset.
        pkg: Package used to generate the pulses (ie where pulse_generation.py is).
        flip_x_y (bool, optional): Whether x and y axis are flipped in the pareto front plot.
            Defaults to True, as normally the activation is the first objective, but we want it in the y axis.
        total_size (Tuple[float, float], optional): This parameter, when multiplied by partial size,
            gives the size of the inset plot (relative to 1). Defaults to (0.5, 0.3).
        partial_size (float, optional): Relative size (within the total size) that the actual inset occupies.
            Any size bigger than 50(%) will result in the pareto point being hidden. Defaults to 40 (%).
        ylim (Tuple[float, float], optional): Limits of the y axis (amplitude) of the inset plot. Defaults to (0, 100).
        x_offset (float, optional): Offset (in the x-axis) for the inset. Between 0 and 1, defaults to 0.
        y_offset (float, optional): Offset (in the x-axis) for the inset. Between 0 and 1, defaults to 0.
    """
    # Create mini-plots for non-dominated solutions
    if flip_x_y:
        ## NOTE: F[0] = activation (in y axis), F[1] = energy (in x axis)
        y, x = pareto_point
    else:
        x, y = pareto_point

    width = ax.get_xlim()[1] - ax.get_xlim()[0]
    height = ax.get_ylim()[1] - ax.get_ylim()[0]
    inset_width = width * total_size[0]
    inset_height = height * total_size[1]
    ## set the inset centered on the point
    x_position = (x - inset_width / 2) / width
    y_position = (y - inset_height / 2) / height
    ## be able to customly move insets around
    x_position += x_offset
    y_position += y_offset
    bbox_to_anchor = [x_position, y_position, total_size[0], total_size[1]]
    # axins = inset_axes(ax, width="40%", height="40%", bbox_to_anchor=bbox_to_anchor, bbox_transform=ax.transAxes)
    axins: Axes = inset_axes(
        ax,
        width=str(partial_size) + "%",
        height=str(partial_size) + "%",
        loc="lower right",  ## must be kept, everything (+- y/x) is relative to this position
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=ax.transAxes,
    )

    # Plot additional details inside the mini-plot if necessary
    pulse = create_pulses_fun(*pulses_args, stds=errors)
    pulse.plot_pulse(axins)
    axins.set_xlim(pulse.time_list[0], pulse.time_list[-1])
    axins.set_title("")
    # axins.set_xlabel("ms")
    # axins.set_label_position("right")
    ## eliminate axis ticks
    # axins.set_xticks([])
    axins.yaxis.tick_right()
    axins.yaxis.set_label_position("right")
    axins.set_ylabel("Ampl (mA)")

    # axins.set_yticks([])
    axins.legend([], [], frameon=False)
    axins.hlines(
        0,
        pulse.time_list[0],
        pulse.time_list[-1],
        color="gray",
        linestyle="--",
        alpha=0.5,
    )

    ## get the position of the upper right corner of the inset plot
    ## taking into account that it is in the LOC=lower right of the bigger plot centered at (x,y)
    x0 = (((50.0 - partial_size) / 100.0) * total_size[0] + x_offset - 0.01) * width
    y0 = ((50.0 - partial_size) / 100.0 * total_size[1] - y_offset - 0.08) * height

    # Create an arrow from the non-dominated point to the inset plot
    arrow = patches.FancyArrowPatch(
        (x, y),
        (
            (x + x0),
            (y - y0)
            - 0.12 * height,  # for some reason the arrow is always a bit too low
        ),
        color="black",
        mutation_scale=15,
        arrowstyle="->",
        linestyle="--",
        alpha=0.75,
    )
    ax.add_patch(arrow)


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
print("DONE")
