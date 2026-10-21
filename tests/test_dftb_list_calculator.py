import asyncio
import importlib
from functools import wraps
from types import SimpleNamespace

import pytest
from odmantic import ObjectId

from molecular_qm_dftb.models.dftb_input import DftbInput
from molecular_qm_models import Atom, Molecule, MoleculeList
from simstack.core.node_runner import NodeRunner
from simstack.models import DataSet
from simstack.models.parameters import SlurmParameters

module = importlib.import_module("molecular_qm_dftb.nodes.dftb_list_calculator")
mass_module = importlib.import_module("simstack.methods.mass_runner")


def _run_async(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return run


@_run_async
@pytest.mark.parametrize(
    "slurm_values,expected",
    [
        ({"tasks": 4, "tasks_per_node": 4}, 4),
        ({"tasks": 4, "tasks_per_node": 2, "nodes": 2}, 4),
        ({"tasks": 3}, 3),
        ({"nodes": 2, "tasks_per_node": 2}, 4),
        ({}, 1),
        ({"tasks": 1, "tasks_per_node": 4}, 1),
    ],
)
async def test_parent_task_allocation_controls_real_semaphore(
    monkeypatch, slurm_values, expected
):
    active = peak = completed = 0
    gate = asyncio.Event()
    saved = []

    async def calculator(molecule, opts, **kwargs):
        nonlocal active, peak, completed
        active += 1
        peak = max(peak, active)
        if active == expected:
            gate.set()
        try:
            await gate.wait()
            await asyncio.sleep(0)
            completed += 1
            return True
        finally:
            active -= 1

    async def save(dataset, db):
        saved.append(dataset)

    # Only replace external calculation and persistence; use the actual MassRunner
    # and its semaphore to verify overlapping child calls and the upper bound.
    monkeypatch.setattr(module, "dftb_calculator", calculator)
    monkeypatch.setattr(DataSet, "save", save)
    fake_context = SimpleNamespace(db=object())
    monkeypatch.setattr(module, "context", fake_context)
    monkeypatch.setattr(mass_module, "context", fake_context)
    runner = NodeRunner("dftb_list_calculator", ObjectId())
    molecules = MoleculeList()
    for index in range(9):
        molecules.append(
            Molecule(atoms=[
                Atom(element="H", x=float(index), y=0, z=0),
                Atom(element="H", x=10, y=0, z=0),
            ])
        )
    parent = SimpleNamespace(slurm_parameters=SlurmParameters(**slurm_values))
    other = SimpleNamespace(slurm_parameters=SlurmParameters(tasks=1))
    await asyncio.wait_for(
        module.dftb_list_calculator.__wrapped__(
            molecules, DftbInput(), node_runner=runner,
            parent_parameters=parent, parameters=other, arg_hash="list-test",
        ),
        timeout=5,
    )
    assert peak == expected
    assert completed == len(molecules)
    assert len(runner.dataset["results"]) == len(molecules)
    assert "tasks" not in runner.dataset
    assert saved[-1] is runner.dataset


@_run_async
async def test_explicit_parameters_fallback_and_child_failure(monkeypatch):
    async def calculator(molecule, opts, **kwargs):
        await asyncio.sleep(0)
        if molecule.atoms[0].x == 1:
            raise RuntimeError("test failure")
        return True

    async def save(dataset, db):
        pass

    monkeypatch.setattr(module, "dftb_calculator", calculator)
    monkeypatch.setattr(DataSet, "save", save)
    fake_context = SimpleNamespace(db=object())
    monkeypatch.setattr(module, "context", fake_context)
    monkeypatch.setattr(mass_module, "context", fake_context)
    runner = NodeRunner("dftb_list_calculator", ObjectId())
    molecules = MoleculeList()
    for index in range(3):
        molecules.append(
            Molecule(atoms=[
                Atom(element="H", x=float(index), y=0, z=0),
                Atom(element="H", x=10, y=0, z=0),
            ])
        )
    await module.dftb_list_calculator.__wrapped__(
        molecules, DftbInput(), node_runner=runner, arg_hash="failure-test",
        parameters=SimpleNamespace(slurm_parameters=SlurmParameters(tasks=2)),
    )
    rows = list(runner.dataset["results"].items())
    assert len(rows) == 3
    assert sum(bool(row["success"].value) for _, row in rows) == 2
    assert any("test failure" in row["error"].value for _, row in rows if "error" in row)


@_run_async
@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({}, "requires kwargs"),
        ({"parent_parameters": SimpleNamespace()}, "slurm_parameters"),
        (
            {"parent_parameters": SimpleNamespace(
                slurm_parameters=SimpleNamespace(tasks=0)
            )},
            "tasks must be positive",
        ),
    ],
)
async def test_missing_or_invalid_allocation_rejected(kwargs, error):
    with pytest.raises(ValueError, match=error):
        await module.dftb_list_calculator.__wrapped__(
            MoleculeList(), DftbInput(),
            node_runner=NodeRunner("dftb_list_calculator", ObjectId()), **kwargs,
        )
