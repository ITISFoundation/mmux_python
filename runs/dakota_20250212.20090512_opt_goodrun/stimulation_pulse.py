"""## _class_ StimulationPulse

Object to load stimulation pulses from txt files.
Tip: Attributes
    - `data` : array with two rows, being the first the time and the second the stimulation amplitude
    - `interpolated_time` : array with the time points where the stimulation amplitude is interpolated
    - `interpolated_pulse_shape` : array with the interpolated stimulation amplitude

Methods:
    - `get_time`() : returns the first row of the `data` attribute
    - `get_pulse_shape` () : returns the second row of the `data` attribute
    - `get_interpolated_time` (dst) : returns the `interpolated_time` attribute (computes it if not previously computed)
    - `get_interpolated_pulse_shape` (dst) : returns the `interpolated_pulse_shape` attribute (computes it if not previously computed)
"""

import os
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


class StimulationPulse:
    def __init__(self, filepath: Optional[Path]):
        """Creates a StimulationPulse object that can be provided to a NEURON simulation or a GAF calculation."""

        if filepath is None:
            self._initialize_stimulus()
            self.name = "Empty Pulse"

    @staticmethod
    def _check_path(filepath: Path) -> Path:
        _, ext = os.path.splitext(os.path.basename(filepath))
        if len(ext) == 0:
            filepath = Path(str(filepath) + ".txt")

        if not os.path.isfile(filepath):
            raise ValueError("File " + str(filepath) + " does not exist.")

        return filepath

    def _load(self, filepath: Path) -> None:
        try:
            data = np.genfromtxt(filepath)
        except Exception:
            raise ValueError("Could not load pulse at " + filepath)

        return data

    def scale_pulse(self, factor: float) -> None:
        self.amplitude_list = [a * factor for a in self.amplitude_list]
        self.name = f"{self.name}_scaled_{factor}"

    def _get_scaled_stim_pulse_path(self, amplitude: float) -> Path:
        """
        Scales the amplitude of the stimulation pulse by a factor and saves it in a new file,
        whose path is returned. The StimulationPulse might be used directly, or the returned path.
        """
        base_path = self.path.parent
        scaled_stim_pulse_name = f"{self.path.stem}_scaled_{amplitude}.txt"
        self.scale_pulse(amplitude)
        self.save_pulse(base_path, scaled_stim_pulse_name, overwrite=True)
        return base_path / scaled_stim_pulse_name

    def get_time(self):
        return self.time_list

    def get_pulse_shape(self):
        return self.amplitude_list

    ## Simply, dst is points per unit of time (in whichever unit that is)
    def get_interpolated_time(self, dst):
        if not hasattr(self, "interpolated_time"):
            t = self.get_time()
            npoints = int(t[-1] * dst + 1)
            assert npoints > 1, "dst is too small"
            self.interpolated_time = np.linspace(t[0], t[-1], npoints)

        return self.interpolated_time

    def get_interpolated_pulse_shape(self, dst):
        if not hasattr(self, "interpolated_pulse_shape"):
            # amp = self.get_pulse_shape()
            # self.interpolated_pulse_shape = interp1d(self.get_interpolated_time(dst), amp)
            interpolator = interp1d(self.get_time(), self.get_pulse_shape())
            self.interpolated_pulse_shape = interpolator(
                self.get_interpolated_time(dst)
            )

        return self.interpolated_pulse_shape

    def plot_pulse(
        self, ax: Optional[plt.Axes] = None, label="Pulse", title=None, show=False
    ):
        # assert hasattr(self, "data"), "No data loaded."

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.time_list, self.amplitude_list, label=label)
        if hasattr(self, "interpolated_time") and hasattr(
            self, "interpolated_pulse_shape"
        ):
            ax.plot(
                self.interpolated_time,
                self.interpolated_pulse_shape,
                label="Interpolated Pulse",
            )
        if hasattr(self, "std_list") and self.std_list is not None:
            # print(len(self.time_list), len(self.amplitude_list), len(self.std_list))
            ax.fill_between(
                self.time_list,
                np.array(self.amplitude_list) - 2 * np.array(self.std_list),
                np.array(self.amplitude_list) + 2 * np.array(self.std_list),
                alpha=0.2,
                label="95% CI",
            )
        else:
            print("No std_list")

        ax.legend()
        title = title if title is not None else self.name
        ax.set_title(title)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (au)")

        if show:
            plt.show()

        return ax

    def save_pulse(self, savedir: Path, filename: str, overwrite: bool = False):
        assert (
            savedir.is_dir()
        ), f"Provided savedir {savedir} is not an existing directory"
        filename = filename if filename.endswith(".txt") else filename + ".txt"
        savepath = savedir / filename
        if savepath.is_file() and not overwrite:
            raise ValueError(
                f"File {savepath} already exists. Enable the 'overwrite' flag to overwrite it."
            )
        np.savetxt(savepath, np.vstack([self.time_list, self.amplitude_list]))

    def _initialize_stimulus(self):
        self.time_list = [0.0]
        self.amplitude_list = [0.0]
        self.std_list = [0.0]
        self.now = 0.0  # variable which will hold last time point

    def _insert_time_interval(
        self,
        amplitude: float,
        time_interval: float,
        std: Optional[float] = None,
    ):
        self.time_list.append(self.now)
        self.amplitude_list.append(amplitude)
        if std is not None:
            self.std_list.append(std)

        #

        self.time_list.append(self.now + time_interval)
        self.amplitude_list.append(amplitude)
        if std is not None:
            self.std_list.append(std)

        self.now += time_interval

    def insert_nonactive_time(self, total_duration):
        """Insert a flat period"""
        starting_time = self.now
        ## this also adds a point of it-is-zero-at-beginning
        self._insert_time_interval(amplitude=0.0, time_interval=total_duration)
        assert (
            self.time_list[-1] == starting_time + total_duration
        )  ## TODO remove after testing
        pass

    def _check_total_duration(self, total_duration):
        self.time_list = [np.round(t, 4) for t in self.time_list]
        assert total_duration >= (self.time_list[-1]), (
            "Please provide a total duration that is longer than the stimulus."
            + f"Currently {total_duration} and {self.time_list[-1]}"
        )

    def finish_pulse(self, total_duration):
        # End of stimulation: amplitude 0.0
        ### make sure that all the stimulation together is still shorter than the total duration
        self._check_total_duration(total_duration)
        ## insert time until end of stimulation at amplitude 0.0
        self._insert_time_interval(amplitude=0.0, time_interval=total_duration, std=0.0)
        self.time_list[-1] = total_duration
        # self._convert_to_numpy()

    def insert_monophasic_square_pulse(
        # def create_monopolar_pulse(
        self,
        amplitude: float = 1.0,
        pulse_width: float = 0.1,
        total_duration: Optional[float] = None,
        initial_delay: float = 0.0,
    ):
        """Creates a monopolar pulse with the specified parameters."""
        if initial_delay > 0.0:
            self._insert_time_interval(amplitude=0.0, time_interval=initial_delay)
        self._insert_time_interval(amplitude, pulse_width)
        if total_duration is not None:
            self._insert_time_interval(
                0.0, total_duration - pulse_width - initial_delay
            )

    def insert_biphasic_square_pulse(self, amplitude, pulse_width, total_duration):
        """Insert a biphasic square pulse, with both phases of same amplitude and pulse_width"""
        self._insert_time_interval(amplitude, pulse_width)
        self._insert_time_interval(-amplitude, pulse_width)
        self._insert_time_interval(
            amplitude=0.0, time_interval=total_duration - 2 * pulse_width
        )
