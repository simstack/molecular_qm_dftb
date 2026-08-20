from molecular_qm_dftb.lib.hsd import build_hsd
from molecular_qm_dftb.models.dftb_input import DftbHamiltonian, DftbInput, SkfSet, XtbMethod
from molecular_qm_models.molecule import Atom, Molecule


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


def test_xtb_hsd_contains_gfn2_and_water_geometry():
    opts = DftbInput(hamiltonian=DftbHamiltonian.XTB, xtb_method=XtbMethod.GFN2)
    hsd = build_hsd(opts, make_water())
    assert "Hamiltonian = xTB" in hsd
    assert 'Method = "GFN2-xTB"' in hsd
    assert "3 C" in hsd
    assert "O H" in hsd
    assert "PrintForces = Yes" in hsd
    assert "ParserVersion = 14" in hsd


def test_dftb3_hsd_has_3ob_hubbard_and_skf_prefix():
    opts = DftbInput(
        hamiltonian=DftbHamiltonian.DFTB,
        skf_set=SkfSet.THREE_OB,
        third_order=True,
        compute_gradients=False,
        compute_cm5=True,
    )
    hsd = build_hsd(opts, make_water())
    assert "Hamiltonian = DFTB" in hsd
    assert "ThirdOrderFull = Yes" in hsd
    assert "H = -0.1857" in hsd
    assert "O = -0.1575" in hsd
    assert 'O = "p"' in hsd
    assert 'H = "s"' in hsd
    assert "3ob-3-1/" in hsd
    assert "CM5 = Yes" in hsd
    assert "PrintForces" not in hsd


def test_optimization_true_does_not_recurse_and_enables_gradients():
    opts = DftbInput(optimization=True, charge=-1, multiplicity=1)
    assert opts.optimization is True
    assert opts.compute_gradients is True
    assert opts.charge == -1
    assert opts.multiplicity == 1


def test_periodic_xtb_writes_lattice_and_kpoints():
    opts = DftbInput(
        use_periodic=True,
        lattice_a=[10.0, 0.0, 0.0],
        lattice_b=[0.0, 10.0, 0.0],
        lattice_c=[0.0, 0.0, 10.0],
        kpoint_mesh=2,
    )
    hsd = build_hsd(opts, make_water())
    assert "3 S" in hsd
    assert "10.0000000000" in hsd
    assert "kPointsAndWeights = SuperCellFolding" in hsd
    assert "2 0 0" in hsd
