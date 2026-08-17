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

    SimstackResult:
        dataset (DataSet): One section named ``results``. Each row has the
            ``dftb_calculator`` node_runner outputs plus the input molecule and
            DftbInput.
    """
    node_runner = kwargs["node_runner"]
    node_runner.info(f"Running DFTB+ on {len(molecules)} molecules")

    async with MassRunner(dftb_calculator, **kwargs) as mass_result:
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
