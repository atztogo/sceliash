"""Calculation of Eliashberg function."""

from __future__ import annotations

from typing import Optional

import numpy as np
from phono3py.other.kaccum import KappaDOSTHM

from .selfen_ph_data import SelfenPH, load_vaspout_h5


class EliashbergFunction:
    """Eliashberg function class.

    Attributes
    ----------
    freq_points : np.ndarray
        Frequency points in 2piTHz unit.
    a2f : np.ndarray
        Eliashberg function in 1/2piTHz unit.
        shape=(ispin, itemp, freq_points)
    temps : np.ndarray
        Temperatures in K.
    lambda_constant : np.ndarray
        Electron-phonon coupling constant.
        shape=(ispin, itemp)

    """

    def __init__(
        self,
        selfen_ph_data: Optional[SelfenPH] = None,
        h5_filename: Optional[str] = None,
    ):
        """Initialize Eliashberg function class.

        Parameters
        ----------
        selfen_ph_data : SelfenPH, optional
            Phonon selfenergy data.
        h5_filename : str, optional
            Path to vaspout.h5 file.

        """
        if selfen_ph_data is None:
            if h5_filename is not None:
                self._data = load_vaspout_h5(h5_filename)
            else:
                raise ValueError("Either data or h5_filename must be provided.")
        else:
            self._data = selfen_ph_data
        self._a2f: Optional[np.ndarray] = None
        self._freq_points: Optional[np.ndarray] = None
        self._lambda_constant: Optional[float] = None

    @property
    def freq_points(self):
        """Return frequency points."""
        return self._freq_points

    @property
    def a2f(self):
        """Return Eliashberg function."""
        return self._a2f

    @property
    def temps(self):
        """Return temperatures."""
        return self._data.temps

    @property
    def frequencies(self):
        """Return frequencies."""
        return self._data.freqs

    @property
    def lambda_constant(self):
        """Return lambda constant."""
        return self._lambda_constant

    def run(self, num_sampling_points: int = 201) -> "EliashbergFunction":
        """Run Eliashberg function calculation."""
        self._freq_points, self._a2f = self._calculate_Eliashberg_function(
            num_sampling_points,
        )
        return self

    def run_lambda_constant(self) -> np.ndarray:
        r"""Compute lambda constant.

        Parameters
        ----------
        freq_points : np.ndarray
            Frequency points in 2piTHz unit.
        a2f : np.ndarray
            Eliashberg function in 1/2piTHz unit.
            shape=(ispin, itemp, freq_points)

        """
        assert np.isclose(
            self._freq_points[1] - self._freq_points[0],
            self._freq_points[-1] - self._freq_points[-2],
        )
        delta_f = self._freq_points[1] - self._freq_points[0]
        indices = np.where(self._freq_points > 1e-5)[0]
        fpts = self._freq_points[indices]

        lambda_vals = np.zeros(self._a2f.shape[:2], dtype="double")
        for i, a2f_spin in enumerate(self._a2f[:, :, indices]):
            for j, a2f_temp in enumerate(a2f_spin):
                lambda_vals[i, j] = (a2f_temp / fpts).sum() * 2 * delta_f

        self._lambda_constant = np.array(lambda_vals)

    def plot_comparison(self):
        """Plot a2F comparison."""
        import matplotlib.pyplot as plt

        plt.figure()
        if len(self._a2f) == 1:
            updown = [""]
        else:
            updown = ["up", "down"]
        for ispin, a2f_spin in enumerate(self._a2f):
            for itemp, temp in enumerate(self._data.temps):
                plt.plot(
                    self._data.freq_points_vasp,
                    self._data.a2f_vasp[ispin, itemp, :],
                    label=f"{temp} K {updown[ispin]} (VASP)",
                )
                plt.plot(
                    self._freq_points,
                    a2f_spin[itemp, :],
                    ".",
                    label=f"{temp} K {updown[ispin]} (phelel)",
                )
        plt.xlabel("Frequency (2piTHz)")
        plt.ylabel("a2F")
        plt.title("a2F vs Frequency")
        plt.legend()
        plt.show()

    def _calculate_Eliashberg_function(
        self,
        num_sampling_points: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate Eliashberg function.

        Returns
        -------
        freq_points : np.ndarray
            Frequency points in 2piTHz unit.
        a2f : np.ndarray
            Eliashberg function in 1/2piTHz unit.
            shape=(ispin, itemp, freq_points)

        """
        # a2f at qj in 1/2piTHz unit [ispin, itemp, ikpt, ib]
        a2f_at_qj = np.zeros(self._data.gamma.shape, dtype="double")
        a2f = []

        # Sum over spins
        dos_at_ef = self._data.dos_at_ef.sum(axis=0)

        for ispin in range(self._data.gamma.shape[0]):
            # delta function is in 1/2piTHz unit.
            for itemp in range(self._data.gamma.shape[1]):
                a2f_at_qj[ispin, itemp, :, :] = (
                    1
                    / (2 * np.pi)
                    * self._data.gamma[ispin, itemp, :, :]
                    / self._data.freqs
                    / dos_at_ef[itemp]
                )

            kappados = KappaDOSTHM(
                a2f_at_qj[ispin, :, :, :, None],
                self._data.freqs,
                self._data.bz_grid,
                ir_grid_points=self._data.ir_grid_points,
                ir_grid_weights=self._data.ir_grid_weights,
                ir_grid_map=self._data.ir_grid_map,
                num_sampling_points=num_sampling_points,
            )
            freq_points, _a2f = kappados.get_kdos()
            a2f.append(_a2f[:, :, 1, 0])

        return freq_points, np.array(a2f)
