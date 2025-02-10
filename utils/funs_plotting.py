import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.axes import Axes


def plot_response_curves(
    fulldata: Dict[str, Dict[str, np.ndarray]],
    output_label: str,
    input_vars: List[str],
    axs: Optional[List[plt.Axes]] = None,  # type: ignore
    savedir: Optional[Path] = None,
    savefmt: str = "png",
):
    if axs is None:
        axs = plt.subplots(
            1, len(input_vars), figsize=(len(input_vars) * 4, 4), sharey=True
        )[1]
        # axs: List[plt.Axes] = axs.flatten()  # type:ignore
    assert axs is not None

    M = 0
    for (var, data), ax in zip(fulldata.items(), axs):
        ax.plot(data["x"], data["y_hat"], label="Predicted")
        if "std_hat" in data:
            ax.fill_between(
                data["x"],
                data["y_hat"] - 2 * data["std_hat"],
                data["y_hat"] + 2 * data["std_hat"],
                alpha=0.3,
            )
        ax.set_xlabel(var)
        m = np.max(data["y_hat"] + 2 * data["std_hat"])
        M = m if M < m else M
    ax.set_ylim(0, M * 1.2)
    plt.suptitle(output_label, fontsize=20)

    if savedir is None:
        savedir = Path(".")

    savepath = savedir / (output_label + "." + savefmt)
    plt.savefig(savepath, format=savefmt, dpi=300)
    print(f"Figure saved in {savepath}")


def plot_objective_space(
    df: pd.DataFrame,
    non_dominated_indices: Optional[List[int]] = None,
    ax: Optional[Axes] = None,  # type: ignore
    xvar: Optional[str] = None,
    yvar: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,  # = (0, 20),
    ylim: Optional[Tuple[float, float]] = None,  # (0, 1e2),
    scattercolor: str = "blue",
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

    xvalues = df[xvar] if xvar else df.iloc[:, 0]
    yvalues = df[yvar] if yvar else df.iloc[:, 1]
    plt.scatter(
        xvalues, yvalues, s=scattersize, facecolors=facecolors, edgecolors=scattercolor
    )
    if non_dominated_indices:
        plt.scatter(
            xvalues.iloc[pd.Index(non_dominated_indices)],
            yvalues.iloc[pd.Index(non_dominated_indices)],
            s=scattersize,
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
