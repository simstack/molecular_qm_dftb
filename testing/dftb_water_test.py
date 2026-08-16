import asyncio

from molecular_qm_dftb import DftbInput, XtbMethod, dftb_calculator
from molecular_qm_models.molecule import Atom, Molecule
from simstack.core.context import context
from simstack.models import Parameters


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


async def main():
    await context.initialize()
    opts = DftbInput(
        molecule=make_water(),
        xtb_method=XtbMethod.GFN2,
        compute_gradients=True,
        compute_charges=True,
    )
    parameters = Parameters(resource="local", in_docker=True, force_rerun=True)
    result = await dftb_calculator(opts, parameters=parameters)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
