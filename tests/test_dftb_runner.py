import pytest

from molecular_qm_dftb.lib.dftb_runner import find_libdftbplus


def test_find_libdftbplus_uses_an_existing_shared_library(tmp_path, monkeypatch):
    library = tmp_path / "libdftbplus.so"
    library.write_bytes(b"")
    monkeypatch.setenv("DFTBPLUS_LIB", str(library))
    assert find_libdftbplus() == str(tmp_path / "libdftbplus")


def test_find_libdftbplus_skips_a_missing_env_path(tmp_path, monkeypatch):
    library_dir = tmp_path / "lib"
    library_dir.mkdir()
    (library_dir / "libdftbplus.so").write_bytes(b"")
    monkeypatch.setenv("DFTBPLUS_LIB", str(tmp_path / "missing" / "libdftbplus"))
    monkeypatch.setattr(
        "molecular_qm_dftb.lib.dftb_runner._LIBRARY_DIRECTORIES",
        (str(library_dir),),
    )
    assert find_libdftbplus() == str(library_dir / "libdftbplus")


def test_find_libdftbplus_raises_when_the_library_is_absent(monkeypatch):
    monkeypatch.delenv("DFTBPLUS_LIB", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr("molecular_qm_dftb.lib.dftb_runner._LIBRARY_DIRECTORIES", ())
    monkeypatch.setattr("molecular_qm_dftb.lib.dftb_runner.sys.prefix", "/no/such/prefix")
    with pytest.raises(RuntimeError, match="libdftbplus was not found"):
        find_libdftbplus()
