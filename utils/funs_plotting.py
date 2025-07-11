import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Callable, Literal
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from funs_data_processing import get_results
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')


def plot_histogram(
    results_dir: Path,
    vars: List[str],
    axs: Optional[List[plt.Axes]] = None,  # type: ignore
    savedir: Optional[Path] = None,
    savename: Optional[str] = None,
    savefmt: str = "png",
    xtick_fmt: str = ".2g",
    fontsize_labels: int = 30,
    fontsize_ticks: int = 18,
    bins: int = 50,
    label_converter: Optional[Callable] = None,  
    title: Optional[str] = None,
):
    if axs is None:
        axs = plt.subplots(
            1, len(vars), figsize=(len(vars) * 4, 4), sharey=True
        )[1]
        # axs: List[plt.Axes] = axs.flatten()  # type:ignore
    assert axs is not None
    
    for var, ax in zip(vars, axs):
        x, xlabel = _process_x_axis_scaling(
            var, {"x": get_results(results_dir / f"predictions.dat", var)}, 
            plotting_xscale="linear",
            label_converter=label_converter
        )
        xlims = (0, np.float64(np.quantile(x, 0.999)))
        ax.hist(x, bins=bins, range=xlims, density=False)
        ax.set_xlabel(xlabel, fontsize=fontsize_labels)
        xticks = np.linspace(xlims[0], xlims[1], 4)
        ax.set_xticks(ticks=xticks, labels=[f"{xt:{xtick_fmt}}" for xt in xticks], fontsize=fontsize_ticks)
        ax.set_xlim(xlims)
    axs[0].set_ylabel("Frequency", fontsize=fontsize_labels)
    axs[0].set_yticklabels(labels=axs[0].get_yticklabels(), fontsize=fontsize_ticks)
    
    if not title:
        title = "Histogram of variables"
    plt.suptitle(title, fontsize=fontsize_labels+4, fontweight="bold")
    
    if savedir is None:
        savedir = Path(".")
    if savename is None:
        savename = "UQ_histogram"
        
    savepath = savedir / (savename + "." + savefmt)
    plt.savefig(savepath, format=savefmt, dpi=600, bbox_inches='tight')  # Ensure full figure is saved
    print(f"Figure saved in {savepath}")


def plot_response_curves(
    fulldata: Dict[str, Dict[str, np.ndarray]],
    response: str,
    input_vars: List[str],
    axs: Optional[List[plt.Axes]] = None,  # type: ignore
    savedir: Optional[Path] = None,
    savename: Optional[str] = None,
    savefmt: str = "png",
    fontsize_labels: int = 30,
    fontsize_ticks: int = 18,
    xtick_fmt: str = ".2g",
    ytick_fmt: str = ".3g",
    plotting_xscale: Literal["linear", "log"] = "linear",
    plotting_yscale: Literal["linear", "log"] = "linear",
    label_converter: Optional[Callable] = None,
    # NB this should remove "log" in any case
):
    if axs is None:
        axs = plt.subplots(
            1, len(input_vars), figsize=(len(input_vars) * 4, 5), sharey=True
        )[1]
        # axs: List[plt.Axes] = axs.flatten()  # type:ignore
    assert axs is not None
    current_yscale = "log" if "log" in response else "linear"

    M = 0
    for var, ax in zip(input_vars, axs):
        assert var in fulldata, f"Variable {var} not found in fulldata"
        data = fulldata[var]
        x, xlabel = _process_x_axis_scaling(
            var, data, plotting_xscale,
            label_converter=label_converter
        )
        y, std, ylabel = _process_y_axis_scaling(
            response, data, plotting_yscale,
            label_converter=label_converter
        )

        ax.plot(x, y, label="Predicted")
        if "std_hat" in data:
            ax.fill_between(x, y - 2 * std, y + 2 * std, alpha=0.3)
        ax.set_xlabel(xlabel, fontsize=fontsize_labels) 
        xticks = np.linspace(np.min(x), np.max(x), 4)
        ax.set_xticks(ticks=xticks, labels=[f"{xt:{xtick_fmt}}" for xt in xticks], fontsize=fontsize_ticks)

        m = np.max(data["y_hat"] + 2 * data["std_hat"])
        M = m if M < m else M

    if current_yscale == "log" and plotting_yscale == "linear":
        ## FIXME current approach, substitute by sth better. This is just changing the labels, not the plot
        yticks = np.linspace(ax.get_ylim()[0]+np.log(1.01), ax.get_ylim()[1] +np.log(0.99), 4)
        axs[0].set_yticks(
            ticks=yticks, labels=[f"{np.exp(y):{ytick_fmt}}" for y in yticks], fontsize=fontsize_ticks
        )
    
    plt.suptitle(ylabel, fontsize=fontsize_labels+4, fontweight="bold")
    # plt.tight_layout()

    if savedir is None:
        savedir = Path(".")
    if savename is None:
        savename = response
        
    savepath = savedir / (savename + "." + savefmt)
    plt.savefig(savepath, format=savefmt, dpi=600, bbox_inches='tight')  # Ensure full figure is saved
    print(f"Figure saved in {savepath}")


from typing import Tuple
def _process_x_axis_scaling(
    var: str,
    data: Dict[str, np.ndarray],
    plotting_xscale: Literal["linear", "log"],
    label_converter: Optional[Callable] = None,
) -> Tuple[np.ndarray, str]:
    """
    Processes the x-axis scaling based on the current and desired plotting scales.

    Args:
        var (str): The variable name indicating the x-axis data.
        data (Dict[str, np.ndarray]): A dictionary containing the data for the variable.
        plotting_xscale (Literal["linear", "log"]): Desired x-axis scale for plotting.
        label_converter (Optional[Callable]): A function to convert variable names into labels.

    Returns:
        Tuple[np.ndarray, str]: A tuple containing the processed x-axis data and the x-axis label.
    """
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
    
    return x, xlabel

def _process_y_axis_scaling(
    response: str,
    data: Dict[str, np.ndarray],
    plotting_yscale: Literal["linear", "log"],
    label_converter: Optional[Callable] = None,
) -> Tuple[np.ndarray, np.ndarray,str]:
    """
    Processes the y-axis scaling based on the current and desired plotting scales.
    Args:
        var (str): The variable name indicating the y-axis data.
        data (Dict[str, np.ndarray]): A dictionary containing the data for the variable.
        plotting_yscale (Literal["linear", "log"]): Desired y-axis scale for plotting.
        label_converter (Optional[Callable]): A function to convert variable names into labels.
    Returns:
        Tuple[np.ndarray, np.ndarray, str]: A tuple containing the processed y-axis data, standard deviation, and the y-axis label.
    """
    current_yscale = "log" if "log" in response else "linear"
    if current_yscale == "log" and plotting_yscale == "log":
        y = data["y_hat"]
        std = data["std_hat"] if "std_hat" in data else np.zeros(len(data["y_hat"]))
        ylabel = (
            "(log) " + label_converter(response)
            if label_converter
            else response
        )
    elif current_yscale == "log" and plotting_yscale == "linear":
        # y = np.exp(data["y_hat"]) ## TODO in prev plots, we didnt do np.exp(y) but rather change ylabels
        y = data["y_hat"]  ## FIXME: current approach: rather change ylabels
        std = (
            data["std_hat"] if "std_hat" in data else np.zeros(len(data["y_hat"]))
        )  ## FIXME how to scale the STD through the log operation??
        ylabel = (
            label_converter(response)
            if label_converter
            else response.split("log_")[-1]
        )
    elif current_yscale == "linear" and plotting_yscale == "linear":
        y = data["y_hat"]
        std = data["std_hat"] if "std_hat" in data else np.zeros(len(data["y_hat"]))
        ylabel = label_converter(response) if label_converter else response
    elif current_yscale == "linear" and plotting_yscale == "log":
        raise NotImplementedError(
            "yscale conversion from linear SuMo to log plotting scale is not implemented"
        )

    return y, std, ylabel


def plot_uq_histogram(
    x: np.ndarray,
    output_response: str,
    ax: Optional[plt.Axes] = None,  # type: ignore
    savedir: Optional[Path] = None,
    savefmt: str = "png",
    plotting_xscale: Literal["linear", "log"] = "linear",
    label_converter: Optional[Callable] = None,
    # NB this should remove "log" in any case
):
    x, xlabel = _process_x_axis_scaling(
        {"x": x}, output_response, plotting_xscale, label_converter
    )
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(x, bins=50, density=False)
    ax.set_xlabel(xlabel)
    if savedir is None:
        savedir = Path(".")
    savepath = savedir / ("UQ_" + output_response + "." + savefmt)
    plt.savefig(savepath, format=savefmt, dpi=300)
    print(f"Figure saved in {savepath}")
    assert savepath is not None
    assert savepath.exists(), f"Plotting failed, savepath {savepath} does not exist"
    return savepath
