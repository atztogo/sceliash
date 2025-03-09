"""Calculation of Eliashberg equation."""

from __future__ import annotations

import numpy as np
from phonopy.units import Hbar, Kb

from .eliashberg_func import EliashbergFunction


class EliashbergEquation:
    """Eliashberg equation class.

    Attributes
    ----------
    freq_points : np.ndarray
        Frequency points in 2piTHz unit.
    a2f : np.ndarray
        Eliashberg function in 1/2piTHz unit. shape=(ispin, itemp, freq_points)
    temps : np.ndarray
        Temperatures in K.
    lambda_function : np.ndarray
        Electron-phonon coupling function. shape=(ispin, itemp, n - n'), where
        -2N <= n - n' <= 2N, and stored in range(0, 2N+2).

    """

    def __init__(
        self,
        eliashberg_func: EliashbergFunction,
    ):
        """Initialize Eliashberg function class.

        Parameters
        ----------
        eliashberg_func : EliashbergFunction
            Eliashberg function class instance.

        """
        self._ef = eliashberg_func

    def run_lambda_function(self) -> np.ndarray:
        r"""Compute lambda function.

        Parameters
        ----------
        freq_points : np.ndarray
            Frequency points in 2piTHz unit.
        a2f : np.ndarray
            Eliashberg function in 1/2piTHz unit.
            shape=(ispin, itemp, freq_points)

        """
        assert np.isclose(
            self._ef.freq_points[1] - self._ef.freq_points[0],
            self._ef.freq_points[-1] - self._ef.freq_points[-2],
        )
        delta_f = self._ef.freq_points[1] - self._ef.freq_points[0]
        indices = np.where(self._ef.freq_points > 1e-5)[0]
        fpts = self._ef.freq_points[indices]

        max_n = np.rint((Hbar / (Kb * self._ef.temps) - np.pi) / (2 * np.pi)).astype(
            int
        )

        for i, a2f_spin in enumerate(self._ef.a2f[:, :, indices]):
            for j, a2f_temp in enumerate(a2f_spin):
                lambda_vals = np.zeros((2 * max_n[j] + 1), dtype="double")
                for _, _ in enumerate(range(-max_n[j], max_n[j] + 1)):
                    lambda_vals[i, j] = (
                        (a2f_temp / fpts).sum() * (2 / len(self._ef.a2f)) * delta_f
                    )

        return np.array(lambda_vals)
