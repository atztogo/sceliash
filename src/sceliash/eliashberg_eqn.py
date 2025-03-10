"""Calculation of Eliashberg equation."""

from __future__ import annotations

from typing import Optional

import numpy as np
from phonopy.units import Kb, THzToEv

from .eliashberg_func import EliashbergFunction


class EliashbergEquation:
    """Eliashberg equation class.

    Attributes
    ----------
    lambda_function : np.ndarray
        Electron-phonon coupling function. shape=(itemp, n - n'), where -N <= n
        <= N, -2N <= n - n' <= 2N, and so values corersponding to n-n' are
        stored in range(0, 2N+2).

    """

    def __init__(self, eliashberg_func: EliashbergFunction, mu_star: float = 0.1):
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
        self._lambda_funcs = self._run_lambda_function()
        return self

    def _run_lambda_function(self) -> list[np.ndarray]:
        """Compute lambda function.

        Returns
        -------
        lambda_funcs : list[np.ndarray]
            List of lambda functions. list[np.ndarray((2N+1, 2N+1))]

        """
        delta_f = self._ef.freq_points[1] - self._ef.freq_points[0]
        indices = np.where(self._ef.freq_points > 1e-5)[0]
        fpts = self._ef.freq_points[indices]  # (freq_points',)
        fpts2 = fpts**2

        omega_c = np.max(self._ef.frequencies) * THzToEv / (2 * np.pi) * 10
        max_n = np.ceil((omega_c / (Kb * self._ef.temps) - np.pi) / (2 * np.pi)).astype(
            int
        )
        lambda_funcs = []
        for i_temp, temp in enumerate(self._ef.temps):
            coef = (2 * np.pi) ** 2 * (Kb * temp) / THzToEv
            N_p = 2 * max_n[i_temp] + 1
            lambda_func = np.zeros(N_p * N_p, dtype="double")
            n_vals = np.arange(N_p) - max_n[i_temp]
            omega_n = coef * n_vals
            a2f_2omega = (
                2 * self._ef.a2f[:, i_temp, indices].sum(axis=0) * delta_f * fpts
            )  # (freq_points',)
            w1_w2_2 = np.subtract.outer(omega_n, omega_n) ** 2
            lambda_func = (a2f_2omega / np.add.outer(w1_w2_2, fpts2)).sum(axis=-1)
            lambda_funcs.append(lambda_func.reshape(N_p, N_p))

        return lambda_funcs

    def run_equation(self) -> np.ndarray:
        """Solve Eliashberg equation.

        Returns
        -------
        max_eigs : np.ndarray
            Maximum eigenvalues of the Eliashberg equation.

        """
        max_eigs = np.zeros(len(self._ef.temps))
        for i, lambda_func in enumerate(self._lambda_funcs):
            gamma_func = lambda_func.copy() - self._mu_star
            N_p = lambda_func.shape[0]
            max_n = (N_p - 1) // 2
            n_vals = np.arange(N_p) - max_n
            p_vals = 2 * n_vals + 1
            v_s = np.sign(p_vals)
            gamma_func[np.diag_indices_from(gamma_func)] -= v_s * (lambda_func @ v_s)
            v_a = np.abs(p_vals)
            gamma_func[:, :] /= np.sqrt(np.outer(v_a, v_a))
            eigvals, _ = np.linalg.eigh(gamma_func)
            max_eigs[i] = np.max(eigvals)

        return max_eigs

    def plot_lambda_function(
        self, temperature: Optional[float] = None, is_log_scale: bool = False
    ):
        """Plot lambda function."""
        import matplotlib.pyplot as plt

        plt.figure()

        for i_temp, temp in enumerate(self._temps):
            if temperature is not None and abs(temp - temperature) > 1e-5:
                continue
            for i_spin, lambda_func_spin in enumerate(self._lambda_funcs[i_temp]):
                N_p = lambda_func_spin.shape[0]
                max_n = (N_p - 1) // 2  # N
                n_vals = np.arange(N_p) - max_n
                lambda_func = np.zeros(4 * max_n + 1, dtype="double")
                for j, n1 in enumerate(n_vals):  # -N .. N
                    for k, n2 in enumerate(n_vals):  # -N .. N
                        if is_log_scale:
                            lambda_func[n1 - n2 + 2 * max_n] = lambda_func_spin[j, k]
                        else:
                            lambda_func[n1 - n2 + 2 * max_n] = (
                                1 / lambda_func_spin[j, k]
                            )
                if is_log_scale:
                    plt.yscale("log")
                plt.plot(
                    np.arange(-2 * max_n, 2 * max_n + 1),
                    lambda_func,
                    label=f"spin={i_spin}, temp={temp}",
                )

        plt.xlabel("n - n'")
        if is_log_scale:
            plt.ylabel("lambda function")
        else:
            plt.ylabel("1/lambda function")
        plt.legend()
        plt.show()
