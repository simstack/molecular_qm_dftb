"""TD-DFTB (Casida) input and EXC.DAT energies for a QMInput."""

import numpy as np

from molecular_qm_dftb.lib.dftb_runner import HARTREE_EV
from molecular_qm_dftb.lib.hsd import build_hsd
from molecular_qm_dftb.models.dftb_input import DftbHamiltonian, DftbInput, SkfSet


def excited_state_hsd(qm_input) -> str:
    """SCC-DFTB2 Casida input for ``qm_input``.

    The Hamiltonian is mio-1-1 SCC-DFTB. Linear response is a second-order
    method, and DFTB+ rejects Casida together with DFTB3. The Fermi
    temperature is 0 K because excited-state forces require integer occupations.
    Closed-shell jobs request singlet excitations. ``basis_set`` and
    ``functional`` are not written; Slater-Koster files replace them.
    """
    if qm_input.states < 1:
        raise ValueError(
            f"QMInput.states must be >= 1 for DFTB excited states, got {qm_input.states}"
        )
    if qm_input.focus_state < 1 or qm_input.focus_state > qm_input.states:
        raise ValueError(
            f"QMInput.focus_state must be between 1 and states={qm_input.states}, "
            f"got {qm_input.focus_state}"
        )
    if qm_input.multiplicity < 1:
        raise ValueError(f"QMInput.multiplicity must be >= 1, got {qm_input.multiplicity}")
    if qm_input.optimization:
        raise ValueError(
            "Set QMInput.optimization=False. This interface returns the "
            "focus_state gradient and does not run a geometry driver."
        )
    if qm_input.use_solvent:
        raise ValueError("Set QMInput.use_solvent=False. DFTB Casida is gas-phase.")
    if qm_input.blocks:
        raise ValueError("Leave QMInput.blocks empty. DFTB Casida has no extra input blocks.")

    opts = DftbInput(
        charge=qm_input.charge,
        multiplicity=qm_input.multiplicity,
        hamiltonian=DftbHamiltonian.DFTB,
        skf_set=SkfSet.MIO,
        third_order=False,
        scc=True,
        compute_gradients=True,
        compute_charges=False,
        compute_cm5=False,
        optimization=False,
        electronic_temperature=0.0,
        max_scc_iterations=qm_input.max_scf_iterations,
    )
    casida = [
        "ExcitedState {",
        "  Casida {",
        f"    NrOfExcitations = {qm_input.states}",
        f"    StateOfInterest = {qm_input.focus_state}",
    ]
    if qm_input.multiplicity == 1:
        casida.append("    Symmetry = Singlet")
    casida.extend(
        [
            "    ExcitedStateForces = Yes",
            "    Diagonaliser = Stratmann {}",
            "  }",
            "}",
        ]
    )
    return build_hsd(opts, qm_input.molecule) + "\n" + "\n".join(casida) + "\n"


def parse_exc_dat(text: str, n_excitations: int) -> np.ndarray:
    """Vertical excitation energies from EXC.DAT, in Hartree, length ``n_excitations``."""
    if n_excitations < 1:
        raise ValueError(f"n_excitations must be >= 1, got {n_excitations}")
    header = None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and set(stripped) <= {"="}:
            header = i
            break
    if header is None:
        raise ValueError("EXC.DAT has no excitation-energy table")
    energies_ev = []
    for line in lines[header + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        token = stripped.split()[0]
        try:
            energies_ev.append(float(token))
        except ValueError as exc:
            raise ValueError(f"EXC.DAT data line does not start with an energy: {stripped}") from exc
    if len(energies_ev) != n_excitations:
        raise ValueError(
            f"EXC.DAT has {len(energies_ev)} excitations, expected {n_excitations}"
        )
    return np.asarray(energies_ev, dtype=np.float64) / HARTREE_EV


def state_total_energies(excitation_hartree: np.ndarray, focus_state: int, focus_total: float) -> np.ndarray:
    """Absolute excited-state energies anchored to the DFTB+ focus-state energy.

    ``excitation_hartree[k]`` is the vertical excitation energy of state ``k + 1``.
    The returned array has the same length. Entry ``focus_state - 1`` is
    ``focus_total`` (the Mermin energy DFTB+ differentiates). Other entries
    keep the EXC.DAT gaps relative to that state.
    """
    if excitation_hartree.ndim != 1 or excitation_hartree.shape[0] < 1:
        raise ValueError(
            f"excitation energies must be a 1-d array with at least one state, "
            f"got shape {excitation_hartree.shape}"
        )
    n_states = int(excitation_hartree.shape[0])
    if focus_state < 1 or focus_state > n_states:
        raise ValueError(f"focus_state must be between 1 and {n_states}, got {focus_state}")
    focus_index = focus_state - 1
    totals = float(focus_total) + (
        np.asarray(excitation_hartree, dtype=np.float64) - float(excitation_hartree[focus_index])
    )
    totals[focus_index] = float(focus_total)
    return totals
