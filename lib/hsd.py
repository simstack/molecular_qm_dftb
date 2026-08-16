"""Build a DFTB+ HSD input for the ctypes Python API."""

from molecular_qm_dftb.lib.skf_params import (
    MIO_MAX_ANGULAR_MOMENTUM,
    THREE_OB_DAMP_XH_EXPONENT,
    THREE_OB_HUBBARD_DERIVS,
    THREE_OB_MAX_ANGULAR_MOMENTUM,
    skf_prefix_for,
)
from molecular_qm_models.molecule import Molecule


def _unique_elements(molecule: Molecule) -> list[str]:
    seen: list[str] = []
    for atom in molecule.atoms:
        element = atom.element
        if element not in seen:
            seen.append(element)
    return seen


def geometry_block(molecule: Molecule, periodic: bool, lattice=None) -> str:
    """Dummy GenFormat geometry. Atom count / species must match ``set_geometry``."""
    if not molecule.atoms:
        raise ValueError("DftbInput.molecule has no atoms")
    elements = _unique_elements(molecule)
    index_of = {el: i + 1 for i, el in enumerate(elements)}
    kind = "S" if periodic else "C"
    lines = [f"{len(molecule.atoms)} {kind}", " ".join(elements)]
    for i, atom in enumerate(molecule.atoms, start=1):
        lines.append(
            f"{i} {index_of[atom.element]} "
            f"{atom.x:16.10f} {atom.y:16.10f} {atom.z:16.10f}"
        )
    if periodic:
        if lattice is None or len(lattice) != 3:
            raise ValueError("periodic geometry requires three lattice vectors")
        lines.append("0.0 0.0 0.0")
        for vec in lattice:
            lines.append(f"{vec[0]:16.10f} {vec[1]:16.10f} {vec[2]:16.10f}")
    return "Geometry = GenFormat {\n" + "\n".join(lines) + "\n}\n"


def _indent_map(mapping: dict[str, object], indent: str = "    ") -> str:
    rows = []
    for key, value in mapping.items():
        if isinstance(value, str):
            rows.append(f'{indent}{key} = "{value}"')
        else:
            rows.append(f"{indent}{key} = {value}")
    return "\n".join(rows)


def _xtb_hamiltonian(opts) -> str:
    charge = opts.charge
    method = opts.xtb_method.value if hasattr(opts.xtb_method, "value") else str(opts.xtb_method)
    lines = [
        "Hamiltonian = xTB {",
        f'  Method = "{method}"',
        f"  Charge = {float(charge)}",
        f"  MaxSCCIterations = {opts.max_scc_iterations}",
        f"  SCCTolerance = {opts.scc_tolerance:.10E}",
        f"  Filling = Fermi {{ Temperature [K] = {opts.electronic_temperature} }}",
    ]
    unpaired = max(opts.multiplicity - 1, 0)
    if unpaired:
        lines.append(f"  SpinPolarisation = Colinear {{ UnpairedElectrons = {unpaired} }}")
    if opts.use_periodic:
        k = opts.kpoint_mesh
        lines.append("  kPointsAndWeights = SuperCellFolding {")
        lines.append(f"    {k} 0 0")
        lines.append(f"    0 {k} 0")
        lines.append(f"    0 0 {k}")
        lines.append("    0.5 0.5 0.5")
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _dftb_hamiltonian(opts, molecule: Molecule) -> str:
    elements = _unique_elements(molecule)
    skf_set = opts.skf_set.value if hasattr(opts.skf_set, "value") else str(opts.skf_set)
    prefix = skf_prefix_for(skf_set, opts.skf_prefix or "")
    if skf_set.startswith("3ob"):
        lmax = THREE_OB_MAX_ANGULAR_MOMENTUM
        hubbard = THREE_OB_HUBBARD_DERIVS
    else:
        lmax = MIO_MAX_ANGULAR_MOMENTUM
        hubbard = {}

    missing = [el for el in elements if el not in lmax]
    if missing:
        raise ValueError(
            f"SKF set {skf_set} has no MaxAngularMomentum for: {', '.join(missing)}"
        )

    lmax_block = _indent_map({el: lmax[el] for el in elements}, indent="    ")
    scc = "Yes" if opts.scc else "No"
    unpaired = max(opts.multiplicity - 1, 0)
    lines = [
        "Hamiltonian = DFTB {",
        f"  Scc = {scc}",
        f"  MaxSCCIterations = {opts.max_scc_iterations}",
        f"  SCCTolerance = {opts.scc_tolerance:.10E}",
        f"  Charge = {float(opts.charge)}",
        f"  Filling = Fermi {{ Temperature [K] = {opts.electronic_temperature} }}",
        "  SlaterKosterFiles = Type2FileNames {",
        f'    Prefix = "{prefix}"',
        '    Separator = "-"',
        '    Suffix = ".skf"',
        "  }",
        "  MaxAngularMomentum {",
        lmax_block,
        "  }",
    ]
    if unpaired:
        lines.append(f"  SpinPolarisation = Colinear {{ UnpairedElectrons = {unpaired} }}")
    if opts.third_order and hubbard:
        hubbard_block = _indent_map({el: hubbard[el] for el in elements}, indent="    ")
        lines.extend(
            [
                "  ThirdOrderFull = Yes",
                "  HubbardDerivs {",
                hubbard_block,
                "  }",
                "  HCorrection = Damping {",
                f"    Exponent = {THREE_OB_DAMP_XH_EXPONENT}",
                "  }",
            ]
        )
    if opts.use_periodic:
        k = opts.kpoint_mesh
        lines.append("  kPointsAndWeights = SuperCellFolding {")
        lines.append(f"    {k} 0 0")
        lines.append(f"    0 {k} 0")
        lines.append(f"    0 0 {k}")
        lines.append("    0.5 0.5 0.5")
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def analysis_block(compute_gradients: bool, compute_cm5: bool) -> str:
    lines = ["Analysis {"]
    if compute_gradients:
        lines.append("  PrintForces = Yes")
    lines.append("  MullikenAnalysis = Yes")
    if compute_cm5:
        lines.append("  CM5 = Yes")
    lines.append("}")
    return "\n".join(lines) + "\n"


def build_hsd(opts) -> str:
    """Return a complete ``dftb_in.hsd`` string for ``DftbInput``."""
    molecule = opts.molecule
    lattice = None
    if opts.use_periodic:
        lattice = [opts.lattice_a, opts.lattice_b, opts.lattice_c]
    hamiltonian_name = (
        opts.hamiltonian.value if hasattr(opts.hamiltonian, "value") else str(opts.hamiltonian)
    )
    parts = [
        geometry_block(molecule, opts.use_periodic, lattice=lattice),
        "",
        _xtb_hamiltonian(opts) if hamiltonian_name == "xTB" else _dftb_hamiltonian(opts, molecule),
        "",
        analysis_block(opts.compute_gradients or opts.optimization, opts.compute_cm5),
        "",
        "Options { WriteDetailedOut = Yes }",
        "",
        f"ParserOptions {{ ParserVersion = {opts.parser_version} }}",
        "",
    ]
    return "\n".join(parts)
