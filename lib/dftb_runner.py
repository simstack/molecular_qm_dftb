"""Run DFTB+ through the dftbplus-python ctypes API."""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from molecular_qm_models.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM
from molecular_qm_models.molecule import Atom, Molecule

try:
    from dftbplus import DftbPlus
except ImportError:
    DftbPlus = None

# DFTB+ pythonapi constants (src/dftbp/common/constants.F90)
HARTREE_EV = 27.2113845
AU_TO_DEBYE = 2.541746473


def find_libdftbplus() -> str:
    """Locate libdftbplus without relying on the pip package's relative path."""
    env = os.environ.get("DFTBPLUS_LIB")
    if env:
        return env
    names = ("libdftbplus.so", "libdftbplus.dylib", "libdftbplus.dll")
    prefixes = [
        os.environ.get("CONDA_PREFIX"),
        "/opt/conda",
        sys.prefix,
    ]
    for prefix in prefixes:
        if not prefix:
            continue
        libdir = Path(prefix) / "lib"
        for name in names:
            candidate = libdir / name
            if candidate.exists() or candidate.is_symlink():
                return str(libdir / "libdftbplus")
    return "libdftbplus"


def molecule_coords_bohr(molecule: Molecule) -> np.ndarray:
    coords = np.array(
        [[atom.x, atom.y, atom.z] for atom in molecule.atoms], dtype=np.float64
    )
    return coords * ANGSTROM_TO_BOHR


def lattice_bohr(lattice_a, lattice_b, lattice_c) -> Optional[np.ndarray]:
    if not lattice_a or not lattice_b or not lattice_c:
        return None
    lat = np.array([lattice_a, lattice_b, lattice_c], dtype=np.float64)
    return lat * ANGSTROM_TO_BOHR


def molecule_from_coords(template: Molecule, coords_bohr: np.ndarray) -> Molecule:
    coords_ang = coords_bohr * BOHR_TO_ANGSTROM
    result = Molecule(
        smiles=template.smiles,
        formula=template.formula,
        properties=dict(template.properties or {}),
    )
    for atom, xyz in zip(template.atoms, coords_ang):
        result.add_atom(
            Atom.from_coords(element=atom.element, coords=[float(xyz[0]), float(xyz[1]), float(xyz[2])])
        )
    return result


def dipole_au(charges: np.ndarray, coords_bohr: np.ndarray) -> np.ndarray:
    return charges.reshape(-1, 1) * coords_bohr


class DftbPlusSession:
    """Thin wrapper around ``dftbplus.DftbPlus`` with Ångström helpers."""

    def __init__(self, hsdpath: str = "dftb_in.hsd", logfile: str = "dftbplus.log"):
        if DftbPlus is None:
            raise RuntimeError("dftbplus-python is not installed in this environment")
        self.calc = DftbPlus(
            libpath=find_libdftbplus(),
            hsdpath=hsdpath,
            logfile=logfile,
        )

    def set_geometry_angstrom(self, coords_angstrom, lattice_angstrom=None) -> None:
        coords = np.ascontiguousarray(np.array(coords_angstrom, dtype=np.float64) * ANGSTROM_TO_BOHR)
        latvecs = None
        if lattice_angstrom is not None:
            latvecs = np.ascontiguousarray(
                np.array(lattice_angstrom, dtype=np.float64) * ANGSTROM_TO_BOHR
            )
        self.calc.set_geometry(coords, latvecs=latvecs)

    def set_geometry_bohr(self, coords_bohr, lattice_bohr_vecs=None) -> None:
        coords = np.ascontiguousarray(np.array(coords_bohr, dtype=np.float64))
        latvecs = None
        if lattice_bohr_vecs is not None:
            latvecs = np.ascontiguousarray(np.array(lattice_bohr_vecs, dtype=np.float64))
        self.calc.set_geometry(coords, latvecs=latvecs)

    def set_external_potential(self, extpot, extpotgrad=None) -> None:
        extpot = np.ascontiguousarray(np.array(extpot, dtype=np.float64).reshape(-1))
        grad = None
        if extpotgrad is not None:
            natom = extpot.shape[0]
            grad = np.ascontiguousarray(np.array(extpotgrad, dtype=np.float64).reshape(natom, 3))
        self.calc.set_external_potential(extpot, extpotgrad=grad)

    def get_nr_atoms(self) -> int:
        return int(self.calc.get_nr_atoms())

    def get_energy(self) -> float:
        return float(self.calc.get_energy())

    def get_gradients(self) -> np.ndarray:
        return np.array(self.calc.get_gradients(), dtype=np.float64)

    def get_gross_charges(self) -> np.ndarray:
        return np.array(self.calc.get_gross_charges(), dtype=np.float64)

    def get_cm5_charges(self) -> np.ndarray:
        return np.array(self.calc.get_cm5_charges(), dtype=np.float64)

    def close(self) -> None:
        self.calc.close()
