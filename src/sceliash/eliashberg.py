"""Calculation of Eliashberg function."""

from __future__ import annotations

from typing import Optional

import h5py
import numpy as np
import spglib
from phono3py.other.kaccum import KappaDOSTHM
from phono3py.phonon.grid import BZGrid, get_grid_point_from_address, get_ir_grid_points
from phonopy.units import Hbar, Kb, THzToEv


def compute_Eliashberg_function(h5_filename: str, num_sampling_points: int = 201):
    """Compute Eliashberg function."""
    eliashberg = EliashbergFunction(h5_filename).run()
    eliashberg.compute_lambda()
    return eliashberg


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

    def __init__(self, h5_filename: str):
        """Initialize Eliashberg function class."""
        self._a2f: Optional[np.ndarray] = None
        self._freq_points: Optional[np.ndarray] = None
        self._lattice: Optional[np.ndarray] = None
        self._positions: Optional[np.ndarray] = None
        self._numbers: Optional[np.ndarray] = None
        self._temps: Optional[np.ndarray] = None
        self._dos_at_ef: Optional[np.ndarray] = None
        self._bz_grid: Optional[BZGrid] = None
        self._ir_grid_points: Optional[np.ndarray] = None
        self._ir_grid_weights: Optional[np.ndarray] = None
        self._ir_grid_map: Optional[np.ndarray] = None
        self._a2f_vasp: Optional[np.ndarray] = None
        self._freq_points_vasp: Optional[np.ndarray] = None
        self._freqs: Optional[np.ndarray] = None
        self._gamma: Optional[np.ndarray] = None
        self._ncdij: Optional[int] = None
        self._load_data(h5_filename)

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
            self._freq_points[1] - self._freq_points[0],
            self._freq_points[-1] - self._freq_points[-2],
        )
        delta_f = self._freq_points[1] - self._freq_points[0]
        indices = np.where(self._freq_points > 1e-5)[0]
        fpts = self._freq_points[indices]

        max_n = np.rint((Hbar / (Kb * self._temps) - np.pi) / (2 * np.pi)).astype(int)

        for i, a2f_spin in enumerate(self._a2f[:, :, indices]):
            for j, a2f_temp in enumerate(a2f_spin):
                lambda_vals = np.zeros((2 * max_n[j] + 1), dtype="double")
                for k, n in enumerate(range(-max_n[j], max_n[j] + 1)):
                    lambda_vals[i, j] = (
                        (a2f_temp / fpts).sum() * (2 / len(self._a2f)) * delta_f
                    )

        return np.array(lambda_vals)

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
            for itemp, temp in enumerate(self._temps):
                plt.plot(
                    self._freq_points_vasp,
                    self._a2f_vasp[ispin, itemp, :],
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

    def _load_data(self, h5_filename: str):
        """Load data from vaspout.h5 file.

        Energy unit is in 2piTHz.

        """
        vals = _collect_data_from_vaspout(h5_filename)
        self._lattice = vals["lattice"]
        self._positions = vals["positions"]
        self._numbers = vals["numbers"]
        k_gen_vecs = vals["k_gen_vecs"]
        ir_kpoints = vals["ir_kpoints"]
        ir_kpoints_weights = vals["ir_kpoints_weights"]
        freqs = vals["freqs"]  # [ikpt, ib]
        self._temps = vals["temps"]
        self._dos_at_ef = vals["dos_at_ef"] * (THzToEv / (2 * np.pi))  # [ispin, itemp]
        gamma = vals["gamma"] / (THzToEv / (2 * np.pi))  # [ispin, ib, ikpt, itemp]
        self._a2f_vasp = vals["a2f_vasp"]
        self._freq_points_vasp = vals["freq_points_vasp"]
        self._ncdij = vals["ncdij"]

        # ir_gps indices in phono3py are mapped to those in vasp.
        sym_dataset = spglib.get_symmetry_dataset(
            (self._lattice, self._positions, self._numbers)
        )
        mesh = np.linalg.inv(self._lattice.T @ k_gen_vecs).T
        mesh = np.rint(mesh).astype(int)
        self._bz_grid = BZGrid(
            mesh, lattice=self._lattice, symmetry_dataset=sym_dataset
        )
        self._ir_grid_points, self._ir_grid_weights, self._ir_grid_map = (
            get_ir_grid_points(self._bz_grid)
        )

        ir_addresss = np.rint(ir_kpoints @ mesh).astype(int)
        gps = get_grid_point_from_address(ir_addresss, self._bz_grid.D_diag)
        irgp = self._ir_grid_map[gps]
        id_map = [np.where(irgp == gp)[0][0] for gp in self._ir_grid_points]
        ir_kpoints_weights *= np.linalg.det(mesh)
        assert (np.abs(self._ir_grid_weights - ir_kpoints_weights[id_map]) < 1e-8).all()

        # Convert kpoints order to that of phono3py using mapping table.
        self._freqs = freqs[id_map]
        self._gamma = gamma[:, :, id_map, :].transpose(0, 3, 2, 1)

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
        a2f_at_qj = np.zeros(self._gamma.shape, dtype="double")
        a2f = []
        for ispin in range(self._gamma.shape[0]):
            # delta function is in 1/2piTHz unit.
            for itemp in range(self._gamma.shape[1]):
                a2f_at_qj[ispin, itemp, :, :] = (
                    1
                    / (2 * np.pi)
                    * self._gamma[ispin, itemp, :, :]
                    / self._freqs
                    / self._dos_at_ef[ispin, itemp]
                )

            kappados = KappaDOSTHM(
                a2f_at_qj[ispin, :, :, :, None],
                self._freqs,
                self._bz_grid,
                ir_grid_points=self._ir_grid_points,
                ir_grid_weights=self._ir_grid_weights,
                ir_grid_map=self._ir_grid_map,
                num_sampling_points=num_sampling_points,
            )
            freq_points, _a2f = kappados.get_kdos()
            a2f.append(_a2f[:, :, 1, 0])

        # non-mag : gamma -> 2gamma
        # collinear : -
        # non-collinear : dos_at_ef -> dos_at_ef / 2
        if self._ncdij == 2:
            coef = 1
        else:
            coef = 2

        return freq_points, np.array(a2f) * coef


def _collect_data_from_vaspout(h5_filename: str) -> dict:
    """Collect data from vaspout.h5 file."""
    vals = {}
    with h5py.File(h5_filename) as f:
        # [itemp]
        vals["temps"] = f["results/electron_phonon/phonons/self_energy_1/temps"][:]
        # [ikpt, 3]
        vals["ir_kpoints"] = f[
            "results/electron_phonon/phonons/self_energy_1/kpoint_coords"
        ][:]
        vals["ir_kpoints_weights"] = f[
            "results/electron_phonon/phonons/self_energy_1/kpoints_symmetry_weight"
        ][:]
        lat_scale = f["results/positions/scale"][()]
        vals["lattice"] = f["results/positions/lattice_vectors"][:] * lat_scale
        number_ion_types = f["results/positions/number_ion_types"][:]
        numbers = []
        for i, nums in enumerate(number_ion_types):
            numbers += [i + 1] * nums
        vals["numbers"] = np.array(numbers)
        vals["positions"] = f["results/positions/position_ions"][:]
        # row vectors in python (and KPOINTS file, too)
        vals["k_gen_vecs"] = f[
            "results/electron_phonon/phonons/self_energy_1/kpoint_generating_vectors"
        ][:]
        # [ispin, itemp] in 1/eV
        vals["dos_at_ef"] = f[
            "results/electron_phonon/phonons/self_energy_1/dos_at_ef"
        ][:, :]
        # [ispin, ib, ikpt , 1, temp, (re,im)] in eV
        vals["gamma"] = (
            -2
            * f["results/electron_phonon/phonons/self_energy_1/selfen_ph"][
                :, :, :, 0, :, 1
            ]
        )
        # [ikpt, ib] in 2piTHz
        vals["freqs"] = f[
            "results/electron_phonon/phonons/self_energy_1/phonon_freqs_ibz"
        ][:]
        # [ispin, itemp, freq_points]
        vals["a2f_vasp"] = f["results/electron_phonon/phonons/self_energy_1/a2F"][:]
        # in 2piTHz
        vals["freq_points_vasp"] = f[
            "results/electron_phonon/phonons/self_energy_1/frequency_grid"
        ][:]
        vals["ncdij"] = f["results/electron_phonon/electrons/dos/ncdij"][()]

    return vals
