import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.axes import Axes
import seaborn as sns

from typing import Callable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches


def plot_response_curves(
    fulldata: Dict[str, Dict[str, np.ndarray]],
    output_label: str,
    input_vars: List[str],
    axs: Optional[List[plt.Axes]] = None,  # type: ignore
    savedir: Optional[Path] = None,
    savefmt: str = "png",
    plotting_xscale: Literal["linear", "log"] = "linear",
    plotting_yscale: Literal["linear", "log"] = "linear",
    label_converter: Optional[
        Callable
    ] = None,  # NB this should remove "log" in any case
):
    if axs is None:
        axs = plt.subplots(
            1, len(input_vars), figsize=(len(input_vars) * 4, 5), sharey=True
        )[1]
        # axs: List[plt.Axes] = axs.flatten()  # type:ignore
    assert axs is not None
    current_yscale = "log" if "log" in output_label else "linear"

    M = 0
    for (var, data), ax in zip(fulldata.items(), axs):
        current_xscale = "log" if "log" in var else "linear"
        if current_xscale == "log" and plotting_xscale == "log":
            x = data["x"]
            xlabel = "(log) " + label_converter(var) if label_converter else var
        elif current_xscale == "log" and plotting_xscale == "linear":
            x = np.exp(data["x"])
            xlabel = label_converter(var) if label_converter else var.split("log_")[-1]
        elif current_xscale == "linear" and plotting_xscale == "log":
            x = np.log(data["x"])
            xlabel = (
                "(log) " + label_converter(var)
                if label_converter
                else "(log) " + var.split("log_")[-1]
            )
        elif current_xscale == "linear" and plotting_xscale == "linear":
            x = data["x"]
            xlabel = label_converter(var) if label_converter else var

        if current_yscale == "log" and plotting_yscale == "log":
            y = data["y_hat"]
            std = data["std_hat"] if "std_hat" in data else np.zeros(len(data["y_hat"]))
            ylabel = (
                "(log) " + label_converter(output_label)
                if label_converter
                else output_label
            )
        elif current_yscale == "log" and plotting_yscale == "linear":
            # y = np.exp(data["y_hat"]) ## TODO in prev plots, we didnt do np.exp(y) but rather change ylabels
            y = data["y_hat"]  ## FIXME: current approach: rather change ylabels
            std = (
                data["std_hat"] if "std_hat" in data else np.zeros(len(data["y_hat"]))
            )  ## FIXME how to scale the STD through the log operation??
            ylabel = (
                label_converter(output_label)
                if label_converter
                else output_label.split("log_")[-1]
            )
        elif current_yscale == "linear" and plotting_yscale == "log":
            raise NotImplementedError(
                "yscale conversion from linear SuMo to log plotting scale is not implemented"
            )
        elif current_yscale == "linear" and plotting_yscale == "linear":
            y = data["y_hat"]
            std = data["std_hat"] if "std_hat" in data else np.zeros(len(data["y_hat"]))
            ylabel = label_converter(output_label) if label_converter else output_label

        ax.plot(x, y, label="Predicted")
        if "std_hat" in data:
            ax.fill_between(x, y - 2 * std, y + 2 * std, alpha=0.3)
        ax.set_xlabel(xlabel, fontsize=14)

        m = np.max(data["y_hat"] + 2 * data["std_hat"])
        M = m if M < m else M
    # ax.set_ylim(0, M * 1.2)
    if current_yscale == "log" and plotting_yscale == "linear":
        ## FIXME current approach, substitute by sth better. This is just changing the labels, not the plot
        axs[0].set_yticks(
            ticks=ax.get_yticks(),
            labels=[f"{np.exp(y):.2e}" for y in ax.get_yticks()],
        )
    plt.suptitle(ylabel, fontsize=20)
    # plt.tight_layout()

    if savedir is None:
        savedir = Path(".")

    savepath = savedir / (output_label + "." + savefmt)
    plt.savefig(savepath, format=savefmt, dpi=300)
    print(f"Figure saved in {savepath}")


def plot_objective_space(
    df: pd.DataFrame,
    xvar: str,
    yvar: str,
    non_dominated_indices: Optional[List[int]] = None,
    ax: Optional[Axes] = None,  # type: ignore
    hvar: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,  # = (0, 20),
    ylim: Optional[Tuple[float, float]] = None,  # (0, 1e2),
    scattercolor: str = "blue",
    palette: str = "Blues",
    scattersize: int = 30,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    facecolors: Optional[str] = None,
    savedir: Optional[Path] = None,
    savefmt: str = "png",
) -> Axes:
    """Plot the objective space of a set of points F."""
    if ax is None:
        ax = plt.subplots(figsize=(10, 10))[1]

    sns.scatterplot(
        df,
        x=xvar,
        y=yvar,
        hue=hvar,
        palette=palette,
        size=scattersize,
        facecolors=facecolors,
        edgecolors=scattercolor,
    )
    if non_dominated_indices:
        sns.scatterplot(
            df.iloc[pd.Index(non_dominated_indices)],
            x=xvar,
            y=yvar,
            size=scattersize,
            facecolors="none",
            edgecolors="red",
        )
    plt.xlabel(xlabel if xlabel else (xvar if xvar else df.columns[0]))
    plt.ylabel(ylabel if ylabel else (yvar if yvar else df.columns[1]))
    plt.xlim(xlim)  # WHAT IF NONE?
    plt.ylim(ylim)

    # plt.hlines(0, xmin=0, xmax=2000, color="gray", linestyle="--", alpha=0.5)

    plt.title(title if title else "Objective Space")
    if savedir is None:
        savedir = Path(".")
    savepath = savedir / (f"ObjectiveSpace_{xvar}_{yvar}." + savefmt)
    plt.savefig(savepath, format=savefmt, dpi=300)
    return ax


def plot_optimization_evolution(
    df: pd.DataFrame,
    output_vars: List[str],
    window: int = 101,  ## window of moving average in the plot
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

    for i, var in enumerate(output_vars):
        ax = axs[i]
        rolling_mean = df[var].rolling(window=window).mean()
        rolling_std = df[var].rolling(window=window).std()
        rolling_min = df[var].rolling(window=window).min()
        rolling_max = df[var].rolling(window=window).max()

        ax.plot(rolling_mean, label=f"{var} (mean)")
        ax.fill_between(
            range(len(df[var])),
            rolling_min,
            rolling_max,
            alpha=0.2,
            label="Spread (min to max)",
        )
        ax.fill_between(
            range(len(df[var])),
            rolling_mean - rolling_std,
            rolling_mean + rolling_std,
            alpha=0.4,
            color="blue",
            label="95% Quantile (mean+-2std)",
        )
        ax.set_xlabel("Evaluation")
        ax.set_title(var)

    ax = axs[len(axs) // 2] if len(axs) % 2 else axs[-1]
    ax.legend()  # make legend only in the middle one, or first one if even
    plt.suptitle("Current optimization evolution", fontsize=14)
    plt.tight_layout()
    if savepath is None:
        if savedir is None:
            savedir = Path(".")
        savepath = savedir / "optimization_evolution.png"
    plt.savefig(savepath, format="png", dpi=300)
    print(f"Figure saved in {savepath}")
    return savepath


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
