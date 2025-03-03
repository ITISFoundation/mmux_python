import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Callable, Literal
import numpy as np
from pathlib import Path

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
    label_converter: Optional[Callable] = None, 
    # NB this should remove "log" in any case
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
        x, xlabel = _process_xscale(data, var, plotting_xscale, label_converter)
        y, std, ylabel = _process_yscale(data, output_label, plotting_yscale, label_converter)

        ax.plot(x, y, label="Predicted")
        if "std_hat" in data:
            ax.fill_between(x, y - 2 * std, y + 2 * std, alpha=0.3)
        ax.set_xlabel(xlabel, fontsize=14)

        m = np.max(data["y_hat"] + 2 * data["std_hat"])
        M = m if M < m else M
        
    if current_yscale == "log" and plotting_yscale == "linear":
        ## FIXME current approach, substitute by sth better. This is just changing the labels, not the plot
        axs[0].set_yticks(
            ticks=ax.get_yticks(),
            labels=[f"{np.exp(y):.2e}" for y in ax.get_yticks()],
        )
    plt.suptitle(ylabel, fontsize=20)

    if savedir is None:
        savedir = Path(".")

    savepath = savedir / (output_label + "." + savefmt)
    plt.savefig(savepath, format=savefmt, dpi=300)
    print(f"Figure saved in {savepath}")
    return savepath

def _process_xscale(data: Dict[str, np.ndarray], var: str, plotting_xscale: Literal["linear", "log"] = "linear", label_converter: Optional[Callable] = None):
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

def _process_yscale(data: Dict[str, np.ndarray], output_label: str, plotting_yscale: Literal["linear", "log"] = "linear", label_converter: Optional[Callable] = None):
    current_yscale = "log" if "log" in output_label else "linear"
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
    x, xlabel = _process_xscale({"x": x} , output_response, plotting_xscale, label_converter)
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(x, bins=50, density=False)
    ax.set_xlabel(label_converter(xlabel) if label_converter else xlabel)
    savefmt: str = "png"
    savepath = savedir / ("UQ_" + output_response + "." + savefmt)
    plt.savefig(savepath, format=savefmt, dpi=300)
    print(f"Figure saved in {savepath}")
    assert savepath is not None
    assert savepath.exists(), f"Plotting failed, savepath {savepath} does not exist"
    return savepath