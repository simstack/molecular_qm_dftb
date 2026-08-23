import asyncio
import random

import pytest

from molecular_qm_dftb import DftbInput
from molecular_qm_dftb.nodes.dftb_list_calculator import dftb_list_calculator
from molecular_qm_models.molecule import Atom, Molecule, MoleculeList
from simstack.core.context import context
from simstack.core.node import node
from simstack.models import Parameters, DataSet
from odmantic import ObjectId

def perturb_molecule(molecule: Molecule, rng: random.Random) -> Molecule:
    for atom in molecule.atoms:
        new_coord = [coord + rng.uniform(-0.05, 0.05) for coord in atom.position()]
        atom.position = new_coord
    return molecule

async def dftb_list_calculator_real_node_submission_random_waters() -> None:
    await context.initialize()
    await list_tester()

@node
async def list_tester(molecule: Molecule, opts: DftbInput, **kwargs):

    rng = random.Random(20260823)
    molecules = MoleculeList()
    for _ in range(200):
        new_molecule = molecule.model_copy(update={"id": ObjectId()})
        perturb_molecule(new_molecule, rng)
        molecules.add_molecule(new_molecule)

    result: DataSet = await dftb_list_calculator(
        molecules,
        opts,
        parameters=Parameters(resource="local", in_docker=True, force_rerun=True)
    )

    assert "results" in result
    assert len(result["results"]) == 10


if __name__ == "__main__":
    asyncio.run(dftb_list_calculator_real_node_submission_random_waters())