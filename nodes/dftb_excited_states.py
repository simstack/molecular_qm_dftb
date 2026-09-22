import logging
from pathlib import Path

import numpy as np

from molecular_qm_dftb.lib.dftb_runner import DftbPlusSession, molecule_coords_bohr
from molecular_qm_dftb.lib.excited_states import (
    excited_state_hsd,
    parse_exc_dat,
    state_total_energies,
)
from molecular_qm_models import QMInput
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models.array_storage import ArrayStorage
from simstack.models.files import FileStack

logger = logging.getLogger(__name__)


@node
async def dftb_excited_states(qm_input: QMInput, **kwargs) -> SimstackResult:
    """
    TD-DFTB excited states through the dftbplus-python API.

    ``QMInput.states`` is the number of Casida roots (singlets when the
    multiplicity is 1). ``QMInput.focus_state`` selects the state whose
    Cartesian gradient is returned (1 is the lowest excited state). The
    Hamiltonian is mio-1-1 SCC-DFTB at 0 K. ``basis_set`` and ``functional``
    are ignored.

    Parameters:
        qm_input (QMInput): Molecule, charge, multiplicity, states, and focus_state.

    SimstackResult:
        energies (ArrayStorage): Total energy of each excited state in Hartree,
            shape (states,). Index 0 is excited state 1. Entry focus_state - 1
            is the DFTB+ Mermin energy of that state; the other entries keep
            the EXC.DAT excitation gaps relative to it.
        gradients (ArrayStorage): Energy gradient of focus_state in Hartree/Bohr,
            shape (n_atoms, 3). This is the gradient of energies[focus_state - 1].
    """
    node_runner = kwargs["node_runner"]
    molecule = qm_input.molecule
    if molecule is not None and molecule.formula and not getattr(node_runner, "custom_name", None):
        node_runner.custom_name = molecule.formula
    logfile = Path("dftbplus.log")
    hsd_path = Path("dftb_in.hsd")
    exc_path = Path("EXC.DAT")
    session = None
    try:
        hsd_path.write_text(excited_state_hsd(qm_input), encoding="utf-8")
        node_runner.info(
            f"TD-DFTB Casida: states={qm_input.states}, focus_state={qm_input.focus_state}, "
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
        grads = np.asarray(session.get_gradients(), dtype=np.float64)
        if tuple(grads.shape) != (n_atoms, 3):
            raise ValueError(f"DFTB+ gradient shape is {tuple(grads.shape)}, expected ({n_atoms}, 3)")
        if not exc_path.is_file():
            raise ValueError("DFTB+ did not write EXC.DAT")
        excitation = parse_exc_dat(exc_path.read_text(encoding="utf-8"), qm_input.states)
        focus_total = float(session.get_energy())
        totals = state_total_energies(excitation, qm_input.focus_state, focus_total)

        energies = ArrayStorage(name="energies")
        energies.array = totals
        gradients = ArrayStorage(name="gradients")
        gradients.array = grads
        node_runner.energies = energies
        node_runner.gradients = gradients
        node_runner.info(
            f"TD-DFTB focus state {qm_input.focus_state}: E={focus_total:.8f} Ha, "
            f"excitations={qm_input.states}"
        )
        return node_runner.succeed()
    except Exception as exc:
        logger.error("DFTB+ excited-state calculation failed: %s", exc)
        if qm_input.tolerate_failure:
            node_runner.warning(f"DFTB+ excited states failed but failure is tolerated: {exc}")
            return node_runner.succeed()
        return node_runner.fail(f"DFTB+ excited-state calculation failed: {exc}")
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
        if node_runner is not None:
            for path in (hsd_path, exc_path, logfile):
                if path.is_file():
                    node_runner.info_files.append(
                        FileStack.from_local_file(
                            path, in_memory=True, is_hashable=True, secure_source=True
                        )
                    )
