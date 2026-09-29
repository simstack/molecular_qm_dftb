import numpy as np
import pytest

from molecular_qm_dftb.lib.state_coupling import (
    excitation_energies,
    hamiltonian_gradient,
    parse_nacv,
    state_coupling_hsd,
)
from .test_excited_states import qm_input


NACV = """\
   0     1
  1.000000000000E+00  0.000000000000E+00  0.000000000000E+00
  0.000000000000E+00  2.000000000000E+00  0.000000000000E+00
 -1.000000000000E+00  0.000000000000E+00  3.000000000000E+00
   0     2
  0.500000000000E+00  0.000000000000E+00  0.000000000000E+00
  0.000000000000E+00  0.000000000000E+00  0.000000000000E+00
  0.000000000000E+00  0.000000000000E+00 -0.500000000000E+00
   1     2
  4.000000000000E+00  0.000000000000E+00  0.000000000000E+00
  0.000000000000E+00  0.000000000000E+00  0.000000000000E+00
  0.000000000000E+00  0.000000000000E+00  0.000000000000E+00
"""


def test_coupling_hsd_requests_state_couplings_without_a_focus_state():
    hsd = state_coupling_hsd(qm_input(), 0, 2)
    assert "StateCouplings = {0 2}" in hsd
    assert "NrOfExcitations = 3" in hsd
    assert "StateOfInterest" not in hsd
    assert "ExcitedStateForces" not in hsd
    assert "PrintForces" not in hsd
    assert "Symmetry = Singlet" in hsd
    assert "Temperature [K] = 0.0" in hsd
    assert "mio-1-1/" in hsd


def test_coupling_hsd_orders_the_pair_and_rejects_a_state_outside_the_roots():
    hsd = state_coupling_hsd(qm_input(), 3, 1)
    assert "StateCouplings = {1 3}" in hsd
    with pytest.raises(ValueError, match="outside"):
        state_coupling_hsd(qm_input(), 0, 4)
    with pytest.raises(ValueError, match="must differ"):
        state_coupling_hsd(qm_input(), 1, 1)


def test_parse_nacv_reads_each_pair():
    pairs = parse_nacv(NACV, 3)
    assert set(pairs) == {(0, 1), (0, 2), (1, 2)}
    assert pairs[(0, 1)][1, 1] == pytest.approx(2.0)
    assert pairs[(1, 2)][0, 0] == pytest.approx(4.0)


def test_hamiltonian_gradient_is_gap_times_coupling_and_symmetric():
    pairs = parse_nacv(NACV, 3)
    energies = excitation_energies(np.array([0.2, 0.5, 0.9]))
    forward = hamiltonian_gradient(pairs, energies, 0, 1)
    backward = hamiltonian_gradient(pairs, energies, 1, 0)
    assert forward == pytest.approx(0.2 * pairs[(0, 1)])
    assert backward == pytest.approx(forward)


def test_excited_pair_uses_the_excitation_gap():
    pairs = parse_nacv(NACV, 3)
    energies = excitation_energies(np.array([0.2, 0.5, 0.9]))
    element = hamiltonian_gradient(pairs, energies, 1, 2)
    assert element == pytest.approx((0.5 - 0.2) * pairs[(1, 2)])


def test_degenerate_states_are_rejected():
    pairs = parse_nacv(NACV, 3)
    energies = excitation_energies(np.array([0.2, 0.2, 0.9]))
    with pytest.raises(ValueError, match="degenerate"):
        hamiltonian_gradient(pairs, energies, 1, 2)


def test_missing_pair_is_rejected():
    pairs = parse_nacv(NACV, 3)
    energies = excitation_energies(np.array([0.2, 0.5, 0.9]))
    with pytest.raises(ValueError, match="no coupling"):
        hamiltonian_gradient(pairs, energies, 0, 3)
