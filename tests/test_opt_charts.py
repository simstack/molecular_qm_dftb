import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
from odmantic import ObjectId

from molecular_qm_dftb.nodes.dftb_calculator import _steepest_descent
from simstack.models.charts_artifact import ChartArtifactModel


class FakeDb:
    def __init__(self):
        self.saved = []

    async def save(self, obj):
        self.saved.append(obj)
        return obj


class FakeSession:
    def __init__(self, energies, gradients):
        self.energies = list(energies)
        self.gradients = list(gradients)
        self.calls = 0

    def set_geometry_bohr(self, coords, latvecs):
        return None

    def get_energy(self):
        energy = self.energies[min(self.calls, len(self.energies) - 1)]
        return energy

    def get_gradients(self):
        grads = self.gradients[min(self.calls, len(self.gradients) - 1)]
        self.calls += 1
        return np.array(grads, dtype=np.float64)


def _run_opt(max_steps, force_tol, energies, gradients, db, task_id):
    node_runner = MagicMock()
    node_runner.task_id = str(task_id)
    session = FakeSession(energies, gradients)
    coords = np.zeros((2, 3), dtype=np.float64)
    kwargs = {"node_runner": node_runner, "task_id": str(task_id)}
    with patch("molecular_qm_dftb.nodes.dftb_calculator._get_db", return_value=db):
        result = asyncio.run(
            _steepest_descent(session, coords, None, max_steps, force_tol, kwargs)
        )
    return result, node_runner


def test_opt_writes_two_charts_every_ten_steps_and_at_end():
    task_id = ObjectId()
    db = FakeDb()
    n_steps = 23
    energies = [-18.0 - 0.01 * i for i in range(n_steps + 2)]
    gradients = [np.full((2, 3), 0.05) for _ in range(n_steps + 2)]
    _run_opt(n_steps, 1e-8, energies, gradients, db, task_id)

    assert len(db.saved) >= 6
    assert all(isinstance(chart, ChartArtifactModel) for chart in db.saved)
    assert all(chart.parent_id == task_id for chart in db.saved)

    titles = [chart.title.text for chart in db.saved]
    assert titles.count("DFTB+ optimization energy") == 3
    assert titles.count("DFTB+ optimization gradient norm") == 3

    energy_ids = {chart.id for chart in db.saved if chart.series[0].yKey == "energy"}
    grad_ids = {chart.id for chart in db.saved if chart.series[0].yKey == "grad_norm"}
    assert len(energy_ids) == 1
    assert len(grad_ids) == 1

    last_energy = [c for c in db.saved if c.series[0].yKey == "energy"][-1]
    last_grad = [c for c in db.saved if c.series[0].yKey == "grad_norm"][-1]
    assert [row["step"] for row in last_energy.data][-1] == n_steps + 1
    assert last_grad.data[-1]["grad_norm"] == float(np.linalg.norm(gradients[0]))


def test_opt_writes_charts_when_converged_before_interval():
    task_id = ObjectId()
    db = FakeDb()
    tiny = np.full((2, 3), 1e-10)
    energies = [-19.0, -19.1]
    gradients = [tiny, tiny]
    result, _ = _run_opt(20, 1e-6, energies, gradients, db, task_id)
    assert result[3] is True
    assert len(db.saved) == 2
    assert {chart.series[0].yKey for chart in db.saved} == {"energy", "grad_norm"}
    assert db.saved[0].parent_id == task_id
