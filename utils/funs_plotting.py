import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import numpy as np
from pathlib import Path


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
