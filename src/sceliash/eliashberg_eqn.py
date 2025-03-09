"""Calculation of Eliashberg equation."""

from __future__ import annotations

import numpy as np
from phonopy.units import Hbar, Kb, THzToEv

from .eliashberg_func import EliashbergFunction


def plot_lambda_function(vaspout_h5: str):
    """Plot lambda function."""
    ee = EliashbergEquation(EliashbergFunction(h5_filename=vaspout_h5).run()).run()
    ee.plot_lambda_function()


class EliashbergEquation:
    """Eliashberg equation class.

    Attributes
    ----------
    lambda_function : np.ndarray
        Electron-phonon coupling function. shape=(ispin, itemp, n - n'), where
        -2N <= n - n' <= 2N, and stored in range(0, 2N+2).

    """

    def __init__(
        self,
        eliashberg_func: EliashbergFunction,
        mu_star: float = 0.1 / THzToEv * 2 * np.pi,
    ):
        """Initialize Eliashberg function class.

        Parameters
        ----------
        eliashberg_func : EliashbergFunction
            Eliashberg function class instance.
        mu_star : float, optional
            Coulomb pseudopotential.

        """
        self._ef = eliashberg_func
        self._mu_star = mu_star

    @property
    def lambda_function(self):
        """Return lambda function."""
        return self._lambda_funcs

    def run(self) -> "EliashbergEquation":
        """Run Eliashberg equation."""
        self._lambda_funcs, self._temps = self._run_lambda_function()
        return self

    def _run_lambda_function(self) -> list[np.ndarray]:
        r"""Compute lambda function.

        Returns
        -------
        lambda_funcs : list[np.ndarray]
            List of lambda functions. list[(ispin, 2N+1, 2N+1)]

        """
        delta_f = self._ef.freq_points[1] - self._ef.freq_points[0]
        indices = np.where(self._ef.freq_points > 1e-5)[0]
        fpts = self._ef.freq_points[indices]

        omega_c = np.max(self._ef.frequencies) * THzToEv / (2 * np.pi) * 10
        max_n = np.rint((omega_c / (Kb * self._ef.temps) - np.pi) / (2 * np.pi)).astype(
            int
        )
        temps = []
        lambda_funcs = []
        for i, temp in enumerate(self._ef.temps):
            if max_n[i] < 50:
                continue
            else:
                temps.append(temp)
            coef = np.pi * 2 / (Hbar / (Kb * temp))
            N_p = 2 * max_n[i] + 1
            lambda_func = np.zeros((self._ef.a2f.shape[0], N_p, N_p), dtype="double")
            for j, a2f_spin in enumerate(self._ef.a2f[:, i, indices]):
                for n1, n2 in np.ndindex((N_p, N_p)):
                    print(j, temp, n1, n2)
                    lambda_func[j, n1, n2] = (
                        a2f_spin
                        * delta_f
                        * 2
                        * fpts
                        / ((coef * (n1 - n2)) ** 2 + fpts**2)
                    ).sum()
                lambda_funcs.append(lambda_func)

        return lambda_funcs, temps

    def plot_lambda_function(self, is_log_scale: bool = True):
        """Plot lambda function."""
        import matplotlib.pyplot as plt

        plt.figure()

        for i, temp in enumerate(self._temps):
            for j, lambda_func_spin in enumerate(self._lambda_funcs[i]):
                N_p = lambda_func_spin.shape[0]
                lambda_func = np.zeros(N_p * 2 + 1, dtype="double")
                for n1, n2 in np.ndindex(lambda_func_spin.shape):
                    if is_log_scale:
                        lambda_func[n1 - n2 + N_p] = lambda_func_spin[n1, n2]
                    else:
                        lambda_func[n1 - n2 + N_p] = 1 / lambda_func_spin[n1, n2]
                if is_log_scale:
                    plt.yscale("log")
                plt.plot(
                    np.arange(-N_p, N_p + 1),
                    lambda_func,
                    label=f"spin={j}, temp={temp}",
                )

        plt.xlabel("n - n'")
        plt.ylabel("lambda function")
        plt.legend()
        plt.show()
