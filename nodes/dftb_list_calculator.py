from datetime import datetime

from molecular_qm_dftb.models.dftb_input import DftbInput
from molecular_qm_dftb.nodes.dftb_calculator import dftb_calculator
from molecular_qm_models.molecule import MoleculeList
from simstack.core.context import context
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.methods.mass_runner import MassRunner
from simstack.models import DataSet, DataSetSection, StringData, DataSetMetadata


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

    Concurrency is one calculator per available Slurm task. Use total tasks
    directly; tasks_per_node describes placement, not the number of tasks
    consumed by a calculator. If total tasks is unset, use nodes * tasks_per_node.
    """
    node_runner = kwargs["node_runner"]
    parameters = kwargs.get("parent_parameters")
    if parameters is None:
        parameters = kwargs.get("parameters")
    if parameters is None:
        raise ValueError(
            "dftb_list_calculator requires kwargs['parent_parameters'] "
            "(or kwargs['parameters'])"
        )
    slurm_parameters = getattr(parameters, "slurm_parameters", None)
    if slurm_parameters is None:
        raise ValueError("dftb_list_calculator requires parameters.slurm_parameters")
    fields_set = getattr(slurm_parameters, "model_fields_set", None)
    slurm_tasks = (
        slurm_parameters.tasks
        if fields_set is None or "tasks" in fields_set
        else None
    )
    tasks_per_node = getattr(slurm_parameters, "tasks_per_node", None)
    nodes = getattr(slurm_parameters, "nodes", 1)
    if slurm_tasks is not None:
        max_concurrency = slurm_tasks
    elif tasks_per_node is not None:
        if nodes is None or nodes < 1 or tasks_per_node < 1:
            raise ValueError("slurm_parameters.nodes and tasks_per_node must be positive")
        max_concurrency = nodes * tasks_per_node
    else:
        max_concurrency = 1
    if max_concurrency < 1:
        raise ValueError("slurm_parameters.tasks must be positive")
    node_runner.info(
        f"Running DFTB+ on {len(molecules)} molecules with max_concurrency={max_concurrency} "
        f"(tasks={slurm_tasks}, tasks_per_node={tasks_per_node}, nodes={nodes})"
    )

    async with MassRunner(dftb_calculator, max_concurrency=max_concurrency, **kwargs) as mass_result:
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

    dataset_metadata = DataSetMetadata(field_name="serialdftb_calculator",
                                       data={
                                           "created_at": datetime.now().isoformat()
                                       })

    results = DataSetSection()

    for index, molecule in enumerate(molecules, start=1):
        node_runner.info(f"Running DFTB+ molecule {index}/{total}")
        node_runner.qm_result = None
        calc_result = await dftb_calculator(molecule, opts, **kwargs)

        if isinstance(calc_result, SimstackResult) and hasattr(calc_result, "status"):
            node_runner.info(f" dftb_calculator returned {calc_result.status}")

        row = {
            "input_molecule": molecule,
            "input_opts": opts,
            "message": StringData(value=calc_result.message),
        }
        qm_result = getattr(node_runner, "qm_result", None)
        if qm_result is not None:
            row["result_dftb_result"] = qm_result
        results.add_row(row)

    dataset = DataSet(field_name="serial_dftb_calculator", metadata=dataset_metadata)
    dataset["results"] = results
    await dataset.save(context.db)

    node_runner.dataset = dataset
    node_runner.info(f"Serial DFTB+ list finished with {len(results)} result rows")
    return node_runner.succeed()
