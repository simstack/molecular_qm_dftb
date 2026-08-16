# Molecular QM DFTB+

DFTB+ capabilities for molecular quantum mechanics within the Simstack framework.
The container ships conda-forge `dftbplus=25.1=nompi_*` and `dftbplus-python=25.1`,
and the node drives the ctypes Python API (`dftbplus.DftbPlus`).

## Nodes

- `dftb_calculator` — write `dftb_in.hsd` from `DftbInput`, then call
  `set_geometry`, `get_energy`, `get_gradients`, `get_gross_charges`,
  optional `get_cm5_charges` / `set_external_potential`. Default Hamiltonian
  is GFN2-xTB via tblite (no Slater-Koster files required). DFTB/3ob and
  DFTB/mio are available when SKF files are on `DFTBPLUS_PARAM_DIR`.

## Dual-use

- **Host (`simstack-model`):** not installable — no `pyproject.toml`. Flat tree for
  `create_node_table` / `create_model_table`.
- **Container:** installable — Dockerfile renames `pyproject.docker` → `pyproject.toml`
  and runs `uv pip install .`. Shared deps install from git (see `pyproject.docker`):
  [`molecular_qm_models`](https://github.com/simstack/molecular_qm_models),
  [`simstack`](https://github.com/simstack/simstack) (`fix-git-pull`).

## Local Docker image

Build from the **simstack-model repository root**:

```bash
docker build -t molecular-qm-dftb:latest -f molecular_qm_dftb/Dockerfile .
```

Only `molecular_qm_dftb/` must be present in the build context; models/simstack
are fetched from git.

## Slater-Koster files

The image installs 3ob-3-1 and mio-1-1 under `/opt/dftbplus/params` and sets
`DFTBPLUS_PARAM_DIR`. Those parameter sets are CC-BY-SA; cite the references
in the upstream READMEs when publishing results that use them.
