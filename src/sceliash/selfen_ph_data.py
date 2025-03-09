"""Data class for phonon selfenergy calculation."""

from __future__ import annotations

import dataclasses
from typing import Optional

import h5py
import numpy as np
import spglib
from phono3py.phonon.grid import BZGrid, get_grid_point_from_address, get_ir_grid_points
from phonopy.units import THzToEv


@dataclasses.dataclass
class SelfenPH:
    """Data class for phonon selfenergy calculation."""

    lattice: np.ndarray
    positions: np.ndarray
    numbers: np.ndarray
    temps: np.ndarray
    dos_at_ef: np.ndarray  # [ispin, itemp]
    bz_grid: BZGrid
    ir_grid_points: np.ndarray
    ir_grid_weights: np.ndarray
    ir_grid_map: np.ndarray
    freqs: np.ndarray  # [ir-kpt, ib]
    gamma: np.ndarray  # [ispin, itemp, ir-kpt, ib]
    ncdij: int  # 1 (nonmag), 2 (collinear), or 4 (non-collinear)
    a2f_vasp: Optional[np.ndarray] = None
    freq_points_vasp: Optional[np.ndarray] = None


def load_vaspout_h5(self, h5_filename: str) -> SelfenPH:
    """Load data from vaspout.h5 file.

    Energy unit is in 2piTHz.

    """
    vals = _collect_data_from_vaspout(h5_filename)
    lattice = vals["lattice"]
    positions = vals["positions"]
    numbers = vals["numbers"]
    k_gen_vecs = vals["k_gen_vecs"]
    ir_kpoints = vals["ir_kpoints"]
    ir_kpoints_weights = vals["ir_kpoints_weights"]
    freqs = vals["freqs"]  # [ikpt, ib]
    temps = vals["temps"]
    dos_at_ef = vals["dos_at_ef"] * (THzToEv / (2 * np.pi))  # [ispin, itemp]
    gamma = vals["gamma"] / (THzToEv / (2 * np.pi))  # [ispin, ib, ikpt, itemp]
    a2f_vasp = vals["a2f_vasp"]
    freq_points_vasp = vals["freq_points_vasp"]
    ncdij = vals["ncdij"]

    # ir_gps indices in phono3py are mapped to those in vasp.
    sym_dataset = spglib.get_symmetry_dataset((lattice, positions, numbers))
    mesh = np.linalg.inv(lattice.T @ k_gen_vecs).T
    mesh = np.rint(mesh).astype(int)
    _bz_grid = BZGrid(mesh, lattice=lattice, symmetry_dataset=sym_dataset)
    _ir_grid_points, _ir_grid_weights, _ir_grid_map = get_ir_grid_points(_bz_grid)

    ir_addresss = np.rint(ir_kpoints @ mesh).astype(int)
    gps = get_grid_point_from_address(ir_addresss, _bz_grid.D_diag)
    irgp = _ir_grid_map[gps]
    id_map = [np.where(irgp == gp)[0][0] for gp in _ir_grid_points]
    ir_kpoints_weights *= np.linalg.det(mesh)
    assert (np.abs(_ir_grid_weights - ir_kpoints_weights[id_map]) < 1e-8).all()

    # Convert kpoints order to that of phono3py using mapping table.
    freqs_ordered = freqs[id_map]
    gamma_ordered = gamma[:, :, id_map, :].transpose(
        0, 3, 2, 1
    )  # [ispin, itemp, ikpt, ib]

    return SelfenPH(
        lattice=lattice,
        positions=positions,
        numbers=numbers,
        temps=temps,
        dos_at_ef=dos_at_ef,
        bz_grid=_bz_grid,
        ir_grid_points=_ir_grid_points,
        ir_grid_weights=_ir_grid_weights,
        ir_grid_map=_ir_grid_map,
        a2f_vasp=a2f_vasp,
        freq_points_vasp=freq_points_vasp,
        freqs=freqs_ordered,
        gamma=gamma_ordered,
        ncdij=ncdij,
    )


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
