import pickle
from pathlib import Path

import pytest

from molecular_qm_dftb.lib.dftb_runner import (
    LOG_NAME,
    RESULT_NAME,
    DftbProcessAborted,
    collect_worker_result,
)


def test_collect_worker_result_reads_log_when_process_dies(tmp_path: Path):
    log = (
        "iSCC Total electronic   Diff electronic      SCC error\n"
        "    1   -0.67344353E+02    0.00000000E+00    0.16793244E-02\n"
        "ERROR!\n"
        "-> Failed to open file 'detailed.out' "
        "[(5004) File already opened in another unit]\n"
    )
    (tmp_path / LOG_NAME).write_text(log, encoding="utf-8")

    with pytest.raises(DftbProcessAborted) as excinfo:
        collect_worker_result(tmp_path, exitcode=-6)

    message = str(excinfo.value)
    assert "exit code -6" in message
    assert "Failed to open file 'detailed.out'" in message
    assert excinfo.value.log_text == log


def test_collect_worker_result_prefers_python_error_in_pickle(tmp_path: Path):
    (tmp_path / LOG_NAME).write_text("partial log\n", encoding="utf-8")
    (tmp_path / RESULT_NAME).write_bytes(
        pickle.dumps({"ok": False, "error": "RuntimeError: API atom count 2 does not match molecule (3)"})
    )

    with pytest.raises(DftbProcessAborted) as excinfo:
        collect_worker_result(tmp_path, exitcode=0)

    assert "API atom count 2" in str(excinfo.value)


def test_collect_worker_result_returns_payload_on_success(tmp_path: Path):
    (tmp_path / LOG_NAME).write_text("ok\n", encoding="utf-8")
    (tmp_path / RESULT_NAME).write_bytes(
        pickle.dumps({"ok": True, "energy": -18.5, "n_atoms": 3})
    )

    payload = collect_worker_result(tmp_path, exitcode=0)
    assert payload["energy"] == -18.5
    assert payload["log_text"] == "ok\n"
