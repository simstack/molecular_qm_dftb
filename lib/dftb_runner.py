"""Run DFTB+ through the dftbplus-python ctypes API."""

import multiprocessing
import os
import pickle
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


HSD_NAME = "dftb_in.hsd"
LOG_NAME = "dftbplus.log"
RESULT_NAME = "result.pkl"
_OPT_STEP = 0.2


class DftbProcessAborted(RuntimeError):
    """DFTB+ Fortran ``error stop`` killed the worker, or the worker failed."""

    def __init__(self, exitcode: Optional[int], log_text: str = ""):
        self.exitcode = exitcode
        self.log_text = log_text or ""
        tail = "\n".join(self.log_text.strip().splitlines()[-40:])
        message = f"DFTB+ process aborted (exit code {exitcode})"
        if tail:
            message = f"{message}:\n{tail}"
        super().__init__(message)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def steepest_descent_sync(session, coords, latvecs, max_steps, force_tol):
    """In-process steepest descent. Caller must isolate CWD if concurrent."""
    energy = None
    grads = None
    energy_history = []
    grad_history = []
    coords = np.array(coords, dtype=np.float64, copy=True)

    def record(iteration):
        nonlocal energy, grads
        energy = session.get_energy()
        grads = session.get_gradients()
        max_force = float(np.max(np.linalg.norm(grads, axis=1)))
        energy_history.append({"step": iteration, "energy": float(energy)})
        grad_history.append({"step": iteration, "grad_norm": float(np.linalg.norm(grads))})
        return max_force

    for iteration in range(1, max_steps + 1):
        session.set_geometry_bohr(coords, latvecs)
        max_force = record(iteration)
        if max_force < force_tol:
            return coords, energy, grads, True, energy_history, grad_history
        coords = coords - _OPT_STEP * grads
    session.set_geometry_bohr(coords, latvecs)
    record(max_steps + 1)
    return coords, energy, grads, False, energy_history, grad_history


def collect_worker_result(scratch: Path, exitcode: Optional[int]) -> dict:
    """Load a worker pickle, or raise with the DFTB+ log if the process died."""
    log_text = _read_text(scratch / LOG_NAME)
    result_path = scratch / RESULT_NAME
    if result_path.exists():
        payload = pickle.loads(result_path.read_bytes())
        if not payload.get("ok"):
            error = payload.get("error") or log_text
            raise DftbProcessAborted(exitcode if exitcode else 1, error)
        payload["log_text"] = log_text
        return payload
    raise DftbProcessAborted(exitcode if exitcode is not None else 1, log_text)


def _run_dftb_in_cwd(request: dict) -> dict:
    session = DftbPlusSession(hsdpath=HSD_NAME, logfile=LOG_NAME)
    try:
        n_atoms = session.get_nr_atoms()
        expected = request.get("expected_n_atoms")
        if expected is not None and n_atoms != expected:
            raise RuntimeError(
                f"API atom count {n_atoms} does not match molecule ({expected})"
            )
        coords = np.array(request["coords"], dtype=np.float64, copy=True)
        latvecs = request.get("latvecs")
        if latvecs is not None:
            latvecs = np.array(latvecs, dtype=np.float64)

        extpot = request.get("extpot")
        if extpot is not None:
            session.set_external_potential(extpot, request.get("extpotgrad"))

        energy_history = []
        grad_history = []
        optimized = None
        if request.get("optimization"):
            coords, energy, grads, optimized, energy_history, grad_history = (
                steepest_descent_sync(
                    session,
                    coords,
                    latvecs,
                    request["max_optimization_steps"],
                    request["force_tolerance"],
                )
            )
        else:
            session.set_geometry_bohr(coords, latvecs)
            energy = session.get_energy()
            grads = session.get_gradients() if request.get("compute_gradients") else None

        charges = session.get_gross_charges() if request.get("compute_charges") else None
        cm5 = None
        cm5_warning = None
        if request.get("compute_cm5"):
            try:
                cm5 = session.get_cm5_charges()
            except Exception as exc:
                cm5_warning = str(exc)

        payload = {
            "ok": True,
            "n_atoms": n_atoms,
            "energy": float(energy),
            "coords": coords,
            "grads": grads,
            "charges": charges,
            "cm5": cm5,
            "optimized": optimized,
            "energy_history": energy_history,
            "grad_history": grad_history,
        }
        if cm5_warning:
            payload["cm5_warning"] = cm5_warning
        return payload
    finally:
        try:
            session.close()
        except Exception:
            pass


def dftb_scratch_worker(scratch: str, request: dict) -> None:
    """Spawn target: chdir into ``scratch`` so Fortran files are not shared."""
    scratch_path = Path(scratch)
    os.chdir(scratch_path)
    try:
        payload = _run_dftb_in_cwd(request)
    except Exception as exc:
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    (scratch_path / RESULT_NAME).write_bytes(
        pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    )


def run_dftb_isolated(scratch: Path, request: dict) -> dict:
    """Run one DFTB+ API job in a child process with its own working directory.

    INSTANCE_SAFE_BUILD only removes writable *memory* globals. Fortran still
    opens ``detailed.out`` (and the logfile) by a fixed name in CWD. Two
    concurrent in-process instances therefore hit gfortran 5004 and
    ``error stop``, which kills the Docker PID and bypasses Python ``except``.
    A spawned process with a private CWD isolates those files; abort becomes
    a non-zero exit the parent can report from ``dftbplus.log``.
    """
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=dftb_scratch_worker,
        args=(str(scratch), request),
        daemon=True,
    )
    proc.start()
    proc.join()
    return collect_worker_result(scratch, proc.exitcode)


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
