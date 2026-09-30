import logging
from pathlib import Path

import numpy as np

from molecular_qm_dftb.lib.dftb_runner import DftbPlusSession, molecule_coords_bohr
from molecular_qm_dftb.lib.excited_states import parse_exc_dat
from molecular_qm_dftb.lib.state_coupling import (
    excitation_energies,
    hamiltonian_gradient,
    parse_nacv,
    state_coupling_hsd,
)
from molecular_qm_models import QMInput
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import IntData
from simstack.models.array_storage import ArrayStorage
from simstack.models.files import FileStack

logger = logging.getLogger(__name__)


def _state_index(data: IntData, name: str) -> int:
    value = data.value
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer DFTB state index, got {value!r}")
    return value


@node
async def dftb_hamiltonian_gradient(
    qm_input: QMInput, state_i: IntData, state_j: IntData, **kwargs
) -> SimstackResult:
    """
    ⟨ψ_i|∇H|ψ_j⟩ along the nuclear coordinates from TD-DFTB.

    State 0 is the SCC-DFTB ground state. State k > 0 is Casida root k.
    ``QMInput.states`` is the number of roots and must be at least the higher
    index. The Hamiltonian is mio-1-1 SCC-DFTB at 0 K. Closed shells are
    singlets. DFTB+ writes the nonadiabatic coupling ⟨ψ_m|∇ψ_n⟩; this node
    returns (E_j - E_i) times ⟨ψ_i|∇ψ_j⟩, in Hartree/Bohr. That product is
    symmetric in the two states. The coupling includes the TD-DFTB Pulay terms.

    Parameters:
        qm_input (QMInput): Molecule, charge, multiplicity, and states.
        state_i (IntData): First state. 0 is the ground state.
        state_j (IntData): Second state. Must differ from state_i.

    SimstackResult:
        hamiltonian_gradient (ArrayStorage): ⟨ψ_i|∇H|ψ_j⟩ in Hartree/Bohr, shape (n_atoms, 3).
    """
    node_runner = kwargs["node_runner"]
    molecule = qm_input.molecule
    if molecule is not None and molecule.formula and not getattr(node_runner, "custom_name", None):
        node_runner.custom_name = molecule.formula
    logfile = Path("dftbplus.log")
    hsd_path = Path("dftb_in.hsd")
    exc_path = Path("EXC.DAT")
    nac_path = Path("NACV.DAT")
    session = None
    try:
        i = _state_index(state_i, "state_i")
        j = _state_index(state_j, "state_j")
        hsd_path.write_text(state_coupling_hsd(qm_input, i, j), encoding="utf-8")
        node_runner.info(
            f"TD-DFTB ⟨ψ{i}|∇H|ψ{j}⟩: states={qm_input.states}, "
            f"charge={qm_input.charge}, multiplicity={qm_input.multiplicity}"
        )
        session = DftbPlusSession(hsdpath=str(hsd_path), logfile=str(logfile))
        n_atoms = session.get_nr_atoms()
        if molecule is None or n_atoms != len(molecule.atoms):
            return node_runner.fail(
                f"API atom count {n_atoms} does not match molecule "
                f"({0 if molecule is None else len(molecule.atoms)})"
            )
        session.set_geometry_bohr(molecule_coords_bohr(molecule))
        session.get_energy()
        if not exc_path.is_file():
            raise ValueError("DFTB+ did not write EXC.DAT")
        if not nac_path.is_file():
            raise ValueError("DFTB+ did not write NACV.DAT")
        excitations = parse_exc_dat(exc_path.read_text(encoding="utf-8"), qm_input.states)
        energies = excitation_energies(excitations)
        pairs = parse_nacv(nac_path.read_text(encoding="utf-8"), n_atoms)
        coupling = hamiltonian_gradient(pairs, energies, i, j)
        if tuple(coupling.shape) != (n_atoms, 3):
            raise ValueError(
                f"⟨ψi|∇H|ψj⟩ shape is {tuple(coupling.shape)}, expected ({n_atoms}, 3)"
            )
        stored = ArrayStorage(name="hamiltonian_gradient")
        stored.array = np.asarray(coupling, dtype=np.float64)
        node_runner.hamiltonian_gradient = stored
        node_runner.info(
            f"⟨ψ{i}|∇H|ψ{j}⟩ gap={energies[j] - energies[i]:.8f} Ha, "
            f"norm={float(np.linalg.norm(coupling)):.8e} Ha/Bohr"
        )
        return node_runner.succeed()
    except Exception as exc:
        logger.error("DFTB+ Hamiltonian coupling failed: %s", exc)
        if qm_input.tolerate_failure:
            node_runner.warning(f"DFTB+ Hamiltonian coupling failed but failure is tolerated: {exc}")
            return node_runner.succeed()
        return node_runner.fail(f"DFTB+ Hamiltonian coupling failed: {exc}")
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
        if node_runner is not None:
            for path in (hsd_path, exc_path, nac_path, logfile):
                if path.is_file():
                    node_runner.info_files.append(
                        FileStack.from_local_file(
                            path, in_memory=True, is_hashable=True, secure_source=True
                        )
                    )
