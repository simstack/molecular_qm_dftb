import numpy as np
import pytest
from molecular_qm_models import BasisSet, Functional, QMInput, QMMethod
from molecular_qm_models.molecule import Atom, Molecule

from molecular_qm_dftb.lib.dftb_runner import HARTREE_EV
from molecular_qm_dftb.lib.excited_states import (
    excited_state_hsd,
    parse_exc_dat,
    state_total_energies,
)


def make_water() -> Molecule:
    coords = [
        [0.0, 0.0, 0.1173],
        [0.0, 0.7572, -0.4692],
        [0.0, -0.7572, -0.4692],
    ]
    molecule = Molecule()
    for element, xyz in zip(["O", "H", "H"], coords):
        molecule.add_atom(Atom.from_coords(element=element, coords=xyz))
    molecule.formula = "H2O"
    return molecule


def qm_input(**overrides) -> QMInput:
    data = dict(
        molecule=make_water(),
        basis_set=BasisSet(basis_set="def2-SVP"),
        functional=Functional(functional="B3LYP"),
        method=QMMethod.TDDFT,
        excited_states=True,
        states=3,
        focus_state=2,
        charge=0,
        multiplicity=1,
    )
    data.update(overrides)
    return QMInput(**data)


EXC_DAT = """ w [eV] Osc.Str. Transition Weight KS [eV] Sym.

 =========================================

 5.551 0.5143882 11 -> 12 1.000 4.207 S
 5.592 0.0000000 10 -> 12 1.000 5.592 S
"""


def test_closed_shell_hsd_requests_singlet_casida_forces():
    hsd = excited_state_hsd(qm_input())
    assert "Hamiltonian = DFTB" in hsd
    assert "ThirdOrderFull" not in hsd
    assert "Temperature [K] = 0.0" in hsd
    assert "PrintForces = Yes" in hsd
    assert "NrOfExcitations = 3" in hsd
    assert "StateOfInterest = 2" in hsd
    assert "Symmetry = Singlet" in hsd
    assert "ExcitedStateForces = Yes" in hsd
    assert "Diagonaliser = Stratmann {}" in hsd
    assert "mio-1-1/" in hsd


def test_open_shell_hsd_omits_excitation_symmetry():
    hsd = excited_state_hsd(qm_input(multiplicity=3))
    assert "UnpairedElectrons = 2" in hsd
    assert "Symmetry" not in hsd
    assert "StateOfInterest = 2" in hsd


def test_states_must_be_positive():
    with pytest.raises(ValueError, match="states must be >= 1"):
        excited_state_hsd(qm_input(method=QMMethod.DFT, excited_states=False, states=4))


def test_focus_state_must_lie_in_the_requested_roots():
    with pytest.raises(ValueError, match="focus_state"):
        excited_state_hsd(qm_input(focus_state=4))


def test_optimization_and_solvent_are_rejected():
    with pytest.raises(ValueError, match="optimization"):
        excited_state_hsd(qm_input(optimization=True))
    with pytest.raises(ValueError, match="use_solvent"):
        excited_state_hsd(qm_input(use_solvent=True, solvent="water"))


def test_parse_exc_dat_converts_ev_to_hartree():
    energies = parse_exc_dat(EXC_DAT, 2)
    assert energies.shape == (2,)
    assert energies[0] == pytest.approx(5.551 / HARTREE_EV)
    assert energies[1] == pytest.approx(5.592 / HARTREE_EV)


def test_parse_exc_dat_rejects_a_short_table():
    with pytest.raises(ValueError, match="EXC.DAT has 2 excitations, expected 3"):
        parse_exc_dat(EXC_DAT, 3)


def test_state_total_energies_anchor_the_focus_state():
    omega = np.array([0.1, 0.3, 0.5])
    totals = state_total_energies(omega, focus_state=2, focus_total=-10.0)
    assert totals.tolist() == pytest.approx([-10.2, -10.0, -9.8])
    assert totals[1] == -10.0
