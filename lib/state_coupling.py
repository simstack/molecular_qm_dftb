"""⟨ψ_i|∇H|ψ_j⟩ from DFTB+ Casida nonadiabatic couplings.

State 0 is the ground state. State k > 0 is excited state k in EXC.DAT.
DFTB+ writes ⟨ψ_m|∇ψ_n⟩ to NACV.DAT. For real eigenstates of a Hermitian
Hamiltonian the Hellmann–Feynman relation used by that derivative coupling is

    ⟨ψ_i|∇_R H|ψ_j⟩ = (E_j - E_i) ⟨ψ_i|∇_R ψ_j⟩

The coupling DFTB+ prints already includes the Pulay (overlap-derivative)
terms of the TD-DFTB formulation, so the product is the numerator of that
nonadiabatic coupling, in Hartree/Bohr. It is symmetric in the two states.
"""

import numpy as np

from molecular_qm_dftb.lib.hsd import build_hsd
from molecular_qm_dftb.models.dftb_input import DftbHamiltonian, DftbInput, SkfSet


def state_coupling_hsd(qm_input, state_i: int, state_j: int) -> str:
    """SCC-DFTB2 Casida input that writes NACV.DAT for one pair of states.

    ``state_i`` and ``state_j`` follow DFTB+ ``StateCouplings``: 0 is the
    ground state and k > 0 is the k-th Casida root. The input requests every
    pair from the lower index through the higher one, which is what DFTB+
    writes. ``StateOfInterest`` is omitted; DFTB+ rejects it together with
    ``StateCouplings`` when that keyword is set explicitly.
    """
    if isinstance(state_i, bool) or not isinstance(state_i, int):
        raise ValueError(f"state_i must be an integer, got {state_i!r}")
    if isinstance(state_j, bool) or not isinstance(state_j, int):
        raise ValueError(f"state_j must be an integer, got {state_j!r}")
    if state_i < 0 or state_j < 0:
        raise ValueError(f"state indices must be >= 0, got {state_i} and {state_j}")
    if state_i == state_j:
        raise ValueError(f"the two states must differ, got {state_i}")
    if qm_input.states < 1:
        raise ValueError(
            f"QMInput.states must be >= 1 for DFTB state couplings, got {qm_input.states}"
        )
    highest = max(state_i, state_j)
    if highest > qm_input.states:
        raise ValueError(
            f"state {highest} is outside QMInput.states={qm_input.states} "
            "(0 is the ground state)"
        )
    if qm_input.multiplicity < 1:
        raise ValueError(f"QMInput.multiplicity must be >= 1, got {qm_input.multiplicity}")
    if qm_input.optimization:
        raise ValueError(
            "Set QMInput.optimization=False. State couplings are a single-point property."
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
        compute_gradients=False,
        compute_charges=False,
        compute_cm5=False,
        optimization=False,
        electronic_temperature=0.0,
        max_scc_iterations=qm_input.max_scf_iterations,
    )
    lower, upper = sorted((state_i, state_j))
    casida = [
        "ExcitedState {",
        "  Casida {",
        f"    NrOfExcitations = {qm_input.states}",
        f"    StateCouplings = {{{lower} {upper}}}",
    ]
    if qm_input.multiplicity == 1:
        casida.append("    Symmetry = Singlet")
    casida.extend(
        [
            "    Diagonaliser = Stratmann {}",
            "  }",
            "}",
        ]
    )
    return build_hsd(opts, qm_input.molecule) + "\n" + "\n".join(casida) + "\n"


def parse_nacv(text: str, n_atoms: int) -> dict[tuple[int, int], np.ndarray]:
    """Nonadiabatic couplings from NACV.DAT.

    Each block is the state pair ``m n`` (m < n) followed by ``n_atoms`` lines
    of Cartesian components in 1/Bohr. The returned arrays have shape
    ``(n_atoms, 3)`` and are ⟨ψ_m|∇ψ_n⟩.
    """
    if isinstance(n_atoms, bool) or not isinstance(n_atoms, int) or n_atoms < 1:
        raise ValueError(f"n_atoms must be an integer >= 1, got {n_atoms!r}")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("NACV.DAT is empty")
    pairs: dict[tuple[int, int], np.ndarray] = {}
    cursor = 0
    while cursor < len(lines):
        header = lines[cursor].split()
        if len(header) != 2:
            raise ValueError(f"NACV.DAT pair header must be two integers, got: {lines[cursor]}")
        try:
            state_m = int(header[0])
            state_n = int(header[1])
        except ValueError as exc:
            raise ValueError(f"NACV.DAT pair header must be two integers, got: {lines[cursor]}") from exc
        if state_m < 0 or state_n <= state_m:
            raise ValueError(f"NACV.DAT pair must have 0 <= m < n, got {state_m} {state_n}")
        block = lines[cursor + 1 : cursor + 1 + n_atoms]
        if len(block) != n_atoms:
            raise ValueError(
                f"NACV.DAT pair {state_m} {state_n} has {len(block)} atom lines, expected {n_atoms}"
            )
        rows = []
        for line in block:
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"NACV.DAT atom line must have 3 components, got: {line}")
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError as exc:
                raise ValueError(f"NACV.DAT atom line is not three floats: {line}") from exc
        key = (state_m, state_n)
        if key in pairs:
            raise ValueError(f"NACV.DAT repeats the coupling of states {state_m} and {state_n}")
        pairs[key] = np.asarray(rows, dtype=np.float64)
        cursor += 1 + n_atoms
    return pairs


def excitation_energies(excitation_hartree: np.ndarray) -> np.ndarray:
    """Excitation energies with the ground state at index 0.

    ``excitation_hartree[k]`` is the EXC.DAT energy of Casida state ``k + 1``.
    The returned array has length ``len(excitation_hartree) + 1`` and entry 0 is 0.
    """
    excitations = np.asarray(excitation_hartree, dtype=np.float64)
    if excitations.ndim != 1 or excitations.shape[0] < 1:
        raise ValueError(
            f"excitation energies must be a 1-d array with at least one state, "
            f"got shape {excitations.shape}"
        )
    return np.concatenate([np.zeros(1, dtype=np.float64), excitations])


def hamiltonian_gradient(pairs, energies: np.ndarray, state_i: int, state_j: int) -> np.ndarray:
    """⟨ψ_i|∇H|ψ_j⟩ = (E_j - E_i) ⟨ψ_i|∇ψ_j⟩, shape (n_atoms, 3), Hartree/Bohr."""
    if isinstance(state_i, bool) or not isinstance(state_i, int):
        raise ValueError(f"state_i must be an integer, got {state_i!r}")
    if isinstance(state_j, bool) or not isinstance(state_j, int):
        raise ValueError(f"state_j must be an integer, got {state_j!r}")
    if state_i < 0 or state_j < 0:
        raise ValueError(f"state indices must be >= 0, got {state_i} and {state_j}")
    if state_i == state_j:
        raise ValueError(f"the two states must differ, got {state_i}")
    levels = np.asarray(energies, dtype=np.float64)
    if levels.ndim != 1:
        raise ValueError(f"energies must be a 1-d array, got shape {levels.shape}")
    for state in (state_i, state_j):
        if state >= levels.shape[0]:
            raise ValueError(
                f"state {state} is outside energies of length {levels.shape[0]} "
                "(index 0 is the ground state)"
            )
    energy_i = float(levels[state_i])
    energy_j = float(levels[state_j])
    if energy_i == energy_j:
        raise ValueError(
            f"states {state_i} and {state_j} are degenerate at {energy_i}; "
            "⟨ψi|∇H|ψj⟩ is not the gap times the nonadiabatic coupling"
        )
    if state_i < state_j:
        key = (state_i, state_j)
        # NACV.DAT stores ⟨ψ_i|∇ψ_j⟩
        bra_ket = 1.0
    else:
        key = (state_j, state_i)
        # ⟨ψ_i|∇ψ_j⟩ = -⟨ψ_j|∇ψ_i⟩
        bra_ket = -1.0
    if key not in pairs:
        raise ValueError(f"NACV.DAT has no coupling for states {key[0]} and {key[1]}")
    return (energy_j - energy_i) * (bra_ket * pairs[key])
