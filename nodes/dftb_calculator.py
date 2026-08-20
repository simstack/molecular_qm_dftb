import logging
from pathlib import Path
from shutil import copyfile

import numpy as np
from odmantic import ObjectId

from molecular_qm_dftb.lib.dftb_runner import (
    AU_TO_DEBYE,
    DftbPlusSession,
    dipole_au,
    lattice_bohr,
    molecule_coords_bohr,
    molecule_from_coords,
)
from molecular_qm_dftb.lib.hsd import build_hsd
from molecular_qm_dftb.models.dftb_input import DftbInput
from molecular_qm_models.molecule import Molecule
from molecular_qm_models.qm_result import QMResult
from simstack.core.context import context
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import FloatData, IntData
from simstack.models.charts_artifact import (
    AGChartAxisConfig,
    AGChartTitleConfig,
    AGLineSeriesConfig,
    ChartArtifactModel,
)
from simstack.models.files import FileStack
from simstack.models.simple_table import SimpleTable

logger = logging.getLogger(__name__)

_CHART_INTERVAL = 10


def _charges_table(name, molecule, charges):
    table = SimpleTable(name=name)
    table.add_column("Index", "number")
    table.add_column("Element", "string")
    table.add_column("Charge", "number")
    for i, (atom, charge) in enumerate(zip(molecule.atoms, charges), start=1):
        table.add_row({"Index": i, "Element": atom.element, "Charge": float(charge)})
    return table


def _gradient_table(molecule, gradients):
    table = SimpleTable(name="DFTB+ gradients (Hartree/Bohr)")
    table.add_column("Index", "number")
    table.add_column("Element", "string")
    table.add_column("gx", "number")
    table.add_column("gy", "number")
    table.add_column("gz", "number")
    table.add_column("|g|", "number")
    for i, (atom, grad) in enumerate(zip(molecule.atoms, gradients), start=1):
        table.add_row(
            {
                "Index": i,
                "Element": atom.element,
                "gx": float(grad[0]),
                "gy": float(grad[1]),
                "gz": float(grad[2]),
                "|g|": float(np.linalg.norm(grad)),
            }
        )
    return table


def _write_hsd(opts: DftbInput, molecule: Molecule, node_runner) -> Path:
    hsd_path = Path("dftb_in.hsd")
    if opts.use_hsd_file and opts.hsd_file is not None:
        downloaded = Path(opts.hsd_file.get(local_dir=Path(".")))
        if downloaded.resolve() != hsd_path.resolve():
            if hsd_path.exists():
                hsd_path.unlink()
            copyfile(downloaded, hsd_path)
        node_runner.info(f"Using provided HSD input {opts.hsd_file.name}")
    else:
        hsd_path.write_text(build_hsd(opts, molecule), encoding="utf-8")
        node_runner.info("Generated dftb_in.hsd from DftbInput")
    node_runner.info_files.append(
        FileStack.from_local_file(hsd_path, in_memory=True, is_hashable=True, secure_source=True)
    )
    return hsd_path


def _task_parent_id(kwargs):
    task_id = kwargs.get("task_id") if kwargs else None
    if task_id is None and kwargs:
        task_id = getattr(kwargs.get("node_runner"), "task_id", None)
    if not task_id:
        return None
    try:
        return ObjectId(str(task_id))
    except Exception:
        return None


def _get_db():
    try:
        return context.db
    except RuntimeError:
        return None


def _opt_line_chart(data, y_key, title, y_label, parent_id, existing=None):
    series = AGLineSeriesConfig(
        type="line",
        xKey="step",
        yKey=y_key,
        title=y_label,
        data=data,
        marker={"enabled": False},
    )
    axes = [
        AGChartAxisConfig(type="number", position="bottom", title="Optimization step"),
        AGChartAxisConfig(type="number", position="left", title=y_label),
    ]
    if existing is not None:
        existing.data = data
        existing.title = AGChartTitleConfig(text=title)
        existing.series = [series]
        existing.axes = axes
        existing.parent_id = parent_id
        return existing
    return ChartArtifactModel(
        parent_id=parent_id,
        data=data,
        title=AGChartTitleConfig(text=title),
        series=[series],
        axes=axes,
    )


async def _save_opt_charts(energy_data, grad_data, kwargs, existing=(None, None)):
    node_runner = None if not kwargs else kwargs.get("node_runner")
    parent_id = _task_parent_id(kwargs)
    if parent_id is None:
        if node_runner is not None:
            node_runner.warning("Skipping optimization charts: missing task_id")
        return existing
    db = _get_db()
    if db is None:
        return existing
    energy_chart = _opt_line_chart(
        list(energy_data),
        "energy",
        "DFTB+ optimization energy",
        "Energy (Ha)",
        parent_id,
        existing[0],
    )
    grad_chart = _opt_line_chart(
        list(grad_data),
        "grad_norm",
        "DFTB+ optimization gradient norm",
        "|g| (Ha/Bohr)",
        parent_id,
        existing[1],
    )
    try:
        await db.save(energy_chart)
        await db.save(grad_chart)
    except Exception as exc:
        if node_runner is not None:
            node_runner.warning(f"Failed to store optimization charts: {exc}")
        else:
            logger.warning("Failed to store optimization charts: %s", exc)
        return existing
    return energy_chart, grad_chart


async def _steepest_descent(session, coords, latvecs, max_steps, force_tol, kwargs):
    node_runner = kwargs.get("node_runner")
    step = 0.2
    energy = None
    grads = None
    energy_history = []
    grad_history = []
    charts = (None, None)

    async def record_and_maybe_chart(iteration, *, force_chart=False):
        nonlocal energy, grads, charts
        energy = session.get_energy()
        grads = session.get_gradients()
        max_force = float(np.max(np.linalg.norm(grads, axis=1)))
        grad_norm = float(np.linalg.norm(grads))
        if node_runner is not None:
            node_runner.info(
                f"opt step {iteration}: E={energy:.8f} Ha, max|g|={max_force:.6e} Ha/Bohr"
            )
        energy_history.append({"step": iteration, "energy": float(energy)})
        grad_history.append({"step": iteration, "grad_norm": grad_norm})
        if force_chart or iteration % _CHART_INTERVAL == 0:
            charts = await _save_opt_charts(energy_history, grad_history, kwargs, charts)
        return max_force

    for iteration in range(1, max_steps + 1):
        session.set_geometry_bohr(coords, latvecs)
        max_force = await record_and_maybe_chart(iteration)
        if max_force < force_tol:
            if node_runner is not None:
                node_runner.info(f"Geometry converged in {iteration} steps")
            if iteration % _CHART_INTERVAL != 0:
                charts = await _save_opt_charts(energy_history, grad_history, kwargs, charts)
            return coords, energy, grads, True
        coords = coords - step * grads
    if node_runner is not None:
        node_runner.warning(f"Geometry not converged after {max_steps} steps")
    session.set_geometry_bohr(coords, latvecs)
    await record_and_maybe_chart(max_steps + 1, force_chart=True)
    return coords, energy, grads, False


@node
async def dftb_calculator(molecule: Molecule, opts: DftbInput, **kwargs) -> SimstackResult:
    """
    DFTB+ node using the dftbplus-python ctypes API.

    Parameters:
        molecule (Molecule): Geometry to evaluate.
        opts (DftbInput): Hamiltonian (xTB or DFTB) and API options.

    SimstackResult:
        qm_result (QMResult): Energy, dipole, charges on atoms, final structure.
        n_atoms (IntData): Number of atoms reported by get_nr_atoms().
        energy_hartree (FloatData): Mermin free energy in Hartree.
        mulliken_charges (SimpleTable): Gross / Mulliken charges when requested.
        cm5_charges (SimpleTable): CM5 charges when requested.
        gradients (SimpleTable): Cartesian gradients in Hartree/Bohr when requested.
    """
    node_runner = kwargs["node_runner"]
    logfile = Path("dftbplus.log")
    session = None
    try:
        if opts.use_external_potential:
            natom = len(molecule.atoms)
            if opts.external_potential is None or len(opts.external_potential) != natom:
                return node_runner.fail("external_potential length must match the number of atoms")
            if opts.external_potential_gradient is not None and len(opts.external_potential_gradient) != natom * 3:
                return node_runner.fail("external_potential_gradient must have length 3 * natom")

        _write_hsd(opts, molecule, node_runner)
        session = DftbPlusSession(hsdpath="dftb_in.hsd", logfile=str(logfile))
        n_atoms = session.get_nr_atoms()
        if n_atoms != len(molecule.atoms):
            return node_runner.fail(
                f"API atom count {n_atoms} does not match molecule ({len(molecule.atoms)})"
            )

        coords = molecule_coords_bohr(molecule)
        latvecs = None
        if opts.use_periodic:
            latvecs = lattice_bohr(opts.lattice_a, opts.lattice_b, opts.lattice_c)

        if opts.use_external_potential:
            session.set_external_potential(
                opts.external_potential, opts.external_potential_gradient
            )
            node_runner.info("Applied population-independent external potential")

        optimized = None
        if opts.optimization:
            coords, energy, grads, optimized = await _steepest_descent(
                session,
                coords,
                latvecs,
                opts.max_optimization_steps,
                opts.force_tolerance,
                kwargs,
            )
        else:
            session.set_geometry_bohr(coords, latvecs)
            energy = session.get_energy()
            grads = session.get_gradients() if opts.compute_gradients else None

        charges = session.get_gross_charges() if opts.compute_charges else None
        cm5 = None
        if opts.compute_cm5:
            try:
                cm5 = session.get_cm5_charges()
            except Exception as exc:
                node_runner.warning(f"get_cm5_charges failed: {exc}")

        final_structure = molecule_from_coords(molecule, coords)
        if charges is not None:
            for atom, charge in zip(final_structure.atoms, charges):
                atom.properties["mulliken_charge"] = float(charge)
        if cm5 is not None:
            for atom, charge in zip(final_structure.atoms, cm5):
                atom.properties["cm5_charge"] = float(charge)

        qm_result = QMResult(
            charge=opts.charge,
            final_energy=float(energy),
            energies=[float(energy)],
            scf_converged=True,
            normal_termination=True,
            final_structure=final_structure,
            optimization_converged=optimized if opts.optimization else None,
        )
        if charges is not None:
            dipole_vec = dipole_au(charges, coords).sum(axis=0)
            qm_result.dipole_moment = [float(x) * AU_TO_DEBYE for x in dipole_vec]
            qm_result.dipole = float(np.linalg.norm(dipole_vec) * AU_TO_DEBYE)
            node_runner.mulliken_charges = _charges_table(
                "Mulliken / Gross charges", final_structure, charges
            )
        if cm5 is not None:
            node_runner.cm5_charges = _charges_table("CM5 charges", final_structure, cm5)
        if grads is not None:
            node_runner.gradients = _gradient_table(final_structure, grads)

        node_runner.qm_result = qm_result
        node_runner.n_atoms = IntData(field_name="n_atoms", value=n_atoms)
        node_runner.energy_hartree = FloatData(field_name="energy_hartree", value=float(energy))
        node_runner.info(
            f"DFTB+ energy={energy:.8f} Ha, n_atoms={n_atoms}, hamiltonian={opts.hamiltonian}"
        )
        return node_runner.succeed()
    except Exception as exc:
        logger.error("DFTB+ calculation failed: %s", exc)
        if opts.tolerate_failure:
            node_runner.warning(f"DFTB+ failed but failure is tolerated: {exc}")
            return node_runner.succeed()
        return node_runner.fail(f"DFTB+ execution failed: {exc}")
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
        if node_runner is not None and logfile.exists():
            node_runner.info_files.append(
                FileStack.from_local_file(
                    logfile, in_memory=True, is_hashable=True, secure_source=True
                )
            )
