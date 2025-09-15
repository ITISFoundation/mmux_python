import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple, Callable, Literal
import pandas as pd
import numpy as np
from pathlib import Path

# sudo apt install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super
plt.rcParams["text.usetex"] = True


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
        ax.set_xlabel(xlabel, fontsize=22)

        # m = np.max(data["y_hat"] + 2 * data["std_hat"])
        # M = m if M < m else M
    # ax.set_ylim(0, M * 1.2)
    if current_yscale == "log" and plotting_yscale == "linear":
        ## FIXME current approach, substitute by sth better. This is just changing the labels, not the plot
        axs[0].set_yticks(
            ticks=ax.get_yticks(),
            labels=[f"{np.exp(y):.2e}" for y in ax.get_yticks()],
            fontsize=18,
        )
    for ax in axs:
        ax.set_xticks(
            ticks=ax.get_xticks(),
            labels=ax.get_xticklabels(),  # type: ignore
            # labels=[f"{x:.2f}" for x in ax.get_xticks()],
            fontsize=20,
        )
    plt.suptitle(ylabel, fontsize=26)
    plt.tight_layout()

    if savedir is None:
        savedir = Path(".")

    savepath = savedir / (output_label + "." + savefmt)
    plt.savefig(savepath, format=savefmt, dpi=600)
    print(f"Figure saved in {savepath}")


def plot_objective_space(
    F: Union[pd.DataFrame, np.ndarray],
    ax: Optional[plt.Axes] = None,  # type: ignore
    xlim: Tuple[float, float] = (0, 20),
    ylim: Tuple[float, float] = (0, 1e2),
    color: str = "blue",
    xlabel: str = "Relative Energy (au)",
    ylabel: str = "Activation (%)",
    title: str = "Objective Space",
    facecolors: str = "none",
):
    """Plot the objective space of a set of points F."""
    if isinstance(F, pd.DataFrame):
        F = F.values

    if ax is None:
        ax = plt.subplots(figsize=(10, 10))[1]

    plt.scatter(F[:, 1], F[:, 0], s=30, facecolors=facecolors, edgecolors=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.hlines(0, xmin=0, xmax=2000, color="gray", linestyle="--", alpha=0.5)

    plt.title(title)
