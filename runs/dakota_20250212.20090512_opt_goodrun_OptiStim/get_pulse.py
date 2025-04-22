import os
import sys

sys.path.append(os.path.dirname(__file__))

import numpy as np

## dont wwanna integrate this changes in S4LNF and have to build a new wheel
## therefore, keep using local file
from stimulation_pulse import StimulationPulse

SEGMENT_PW = 15e-3  # Width, in ms, of 1 pulse segment
DST = int(1 / SEGMENT_PW)  # Points per unit of time (ms)
DURATION = 0.24  # Duration of the pulse, in ms
NVARS = int(DURATION / SEGMENT_PW) - 1


def create_zero_mean_projection(xN1):
    """Projects to a one-higher dimension in which the mean is always zero.
    This should help make the inputs more independently meaninginful (instead of substracting the mean, depending on all others).
    However, I fear it might be equivalent to substracting the mean.
    In that case, simply try the alternative, appending a compensating pulse at the end, or simply removing the demeaning condition.
    """

    N = len(xN1) + 1
    ones_N = np.ones((N, 1))

    P = np.eye(N) - 1 / N * (ones_N @ ones_N.T)
    xN = P @ np.hstack([xN1, [0.0]])
    return xN


## alternative option to the projection
def remove_mean(xN1):
    """Removes the mean from the input vector xN1."""
    mean_value = np.mean(xN1)
    xN = np.hstack([xN1 - mean_value, [0.0]])
    return xN


def get_pulse(*args, stds=None) -> StimulationPulse:
    assert len(args) == NVARS, (
        "Number of arguments must match the duration of the pulse."
        + f"Currently {len(args)} and {NVARS}"
    )

    pulse_object = StimulationPulse(None)
    pulse_object.name = "Free Pulse"

    amps = np.array(args)
    current_balanced_amplitudes = create_zero_mean_projection(amps)

    # Create the pulse
    for i, amp in enumerate(current_balanced_amplitudes):
        std = stds[i] if stds is not None else None
        pulse_object._insert_time_interval(amp, SEGMENT_PW, std=std)

    pulse_object.finish_pulse(DURATION)

    return pulse_object
