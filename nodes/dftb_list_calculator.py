from molecular_qm_dftb.models.dftb_input import DftbInput
from molecular_qm_dftb.nodes.dftb_calculator import dftb_calculator
from molecular_qm_models.molecule import MoleculeList
from simstack.core.context import context
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.methods.mass_runner import MassRunner
from simstack.models import DataSet, DataSetSection


@node
async def dftb_list_calculator(
    molecules: MoleculeList, opts: DftbInput, **kwargs
) -> SimstackResult:
    """
    Run ``dftb_calculator`` on every molecule in parallel.

    Parameters:
        molecules (MoleculeList): Molecules to evaluate with the same DftbInput.
        opts (DftbInput): Shared DFTB+/xTB options.

    Called Nodes:
        dftb_calculator

    SimstackResult:
        dataset (DataSet): One section named ``results``. Each row has the
            ``dftb_calculator`` node_runner outputs plus the input molecule and
            DftbInput.
    """
    node_runner = kwargs["node_runner"]
    node_runner.info(f"Running DFTB+ on {len(molecules)} molecules with max_concurrency=5")

    async with MassRunner(dftb_calculator, max_concurrency=5, **kwargs) as mass_result:
        for molecule in molecules:
            mass_result.create_tasks(molecule, opts)

    dataset: DataSet = mass_result.dataset
    tasks = dataset.pop("tasks")
    if tasks is None:
        tasks = DataSetSection()
    dataset["results"] = tasks
    await dataset.save(context.db)

    node_runner.dataset = dataset
    node_runner.info(f"DFTB+ list finished with {len(dataset['results'])} result rows")
    return node_runner.succeed()


@node
async def serial_dftb_list_calculator(
    molecules: MoleculeList, opts: DftbInput, **kwargs
) -> SimstackResult:
    """
    Run ``dftb_calculator`` on every molecule strictly in sequence.

    Parameters:
        molecules (MoleculeList): Molecules to evaluate with the same DftbInput.
        opts (DftbInput): Shared DFTB+/xTB options.

    Called Nodes:
        dftb_calculator

    SimstackResult:
        dataset (DataSet): One section named ``results``. Each row has the
            input molecule, DftbInput, success flag and optional
            ``result_dftb_result`` (QMResult).
    """
    node_runner = kwargs["node_runner"]
    total = len(molecules)
    node_runner.info(f"Running DFTB+ on {total} molecules in sequence")

    results = DataSetSection()

    for index, molecule in enumerate(molecules, start=1):
        node_runner.info(f"Running DFTB+ molecule {index}/{total}")
        node_runner.qm_result = None
        calc_result = await dftb_calculator(molecule, opts, **kwargs)

        row = {
            "input_molecule": molecule,
            "input_opts": opts,
            "success": bool(calc_result.success),
            "message": calc_result.message,
        }
        qm_result = getattr(node_runner, "qm_result", None)
        if qm_result is not None:
            row["result_dftb_result"] = qm_result
        results.append(row)

    dataset = DataSet()
    dataset["results"] = results
    await dataset.save(context.db)

    node_runner.dataset = dataset
    node_runner.info(f"Serial DFTB+ list finished with {len(results)} result rows")
    return node_runner.succeed()
