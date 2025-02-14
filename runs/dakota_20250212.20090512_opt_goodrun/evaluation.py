import os
import sys

sys.path.append(os.path.dirname(__file__))


import numpy as np
from pathlib import Path
from get_pulse import get_pulse
from get_pulse import SEGMENT_PW, NVARS, DST
from s4l_neurofunctions.af.af_data_object import AFDataObject
from s4l_neurofunctions.af.gaf_calculator import GAFCalculatorHomogeneous


def deactivate_tqdm():
    import os

    os.environ["TQDM_DISABLE"] = "1"

    from tqdm import tqdm
    from functools import partialmethod

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # type: ignore


def sigmoid(x: np.ndarray, threshold: float, slope: float) -> np.ndarray:
    return 1 / (1 + np.exp(-slope * (x - threshold)))


deactivate_tqdm()
afdataloaded = AFDataObject(af_type="AF").load(Path(__file__).parent)


def model(**kwargs):
    print(f"Evaluating Free Pulse for {kwargs}")
    params = [k for k in kwargs.values()]
    return {
        "activation": evaluate_activation(params),
        "energy": evaluate_energy(params),
        "maxamp": evaluate_maxamp(params),
    }


model.__annotations__.update({"inputs": {f"p{i+1}": float for i in range(NVARS)}})
model.__annotations__.update(
    {"outputs": {"activation": float, "energy": float, "maxamp": float}}
)  ## clearly custom-made annotations; do for now until we have the proper function database
## Users will still need to define inputs & outputs of their python functions when adding them there


def evaluate_maxamp(x) -> float:
    pulse = get_pulse(*x)
    return np.max(np.abs(pulse.amplitude_list))


def evaluate_activation(x) -> float:
    pulse = get_pulse(*x)

    gafc = GAFCalculatorHomogeneous(dst=DST)
    gafc.compute_gaf(
        afdataloaded,
        pulse,
        force_recomputation=True,
        MODE="time_integration",
    )

    gafpeaks = gafc.get_peaks().get_gaf_data()
    gafmax = gafpeaks.AF_max.values
    act = np.mean(sigmoid(gafmax, threshold=14.0, slope=2.0))  # type: ignore
    return act


def evaluate_energy(x) -> float:
    pulse = get_pulse(*x)
    R = 1.2e3  # 1.2 kOhm -- from the model, +-1V generates 1.65mA
    ## total work = sum I^2 * R * dt
    energy = [(i * 1e-3) ** 2 * R * (SEGMENT_PW * 1e-3) for i in pulse.amplitude_list]
    energy = sum(energy)
    return energy


if __name__ == "__main__":
    print(model.__annotations__)
    print(model(**{f"p{i+1}": np.random.randn() for i in range(NVARS)}))
    print("DONE")
