# Molecular QM DFTB+

DFTB+ capabilities for molecular quantum mechanics within the Simstack framework.
The container ships conda-forge `dftbplus=25.1=nompi_*` and `dftbplus-python=25.1`,
and the node drives the ctypes Python API (`dftbplus.DftbPlus`).

## Nodes

- `dftb_calculator` — `Molecule` + `DftbInput`; write `dftb_in.hsd`, then call
  `set_geometry`, `get_energy`, `get_gradients`, `get_gross_charges`,
  optional `get_cm5_charges` / `set_external_potential`. Default Hamiltonian
  is DFTB with the 3ob-3-1 Slater-Koster set (SKF files on
  `DFTBPLUS_PARAM_DIR`). xTB via tblite (GFN1/GFN2/IPEA1) is available as an
  alternative.
- `dftb_list_calculator` — `MoleculeList` + `DftbInput`; runs `dftb_calculator`
  on every molecule in parallel and returns a `DataSet` with one section
  named `results`.

## Dual-use

- **Host (`simstack-model`):** not installable — no `pyproject.toml`. Flat tree for
  `create_node_table` / `create_model_table`.
- **Container:** installable — Dockerfile renames `pyproject.docker` → `pyproject.toml`
  and runs `uv pip install .`. Shared deps install from git (see `pyproject.docker`):
  [`molecular_qm_models`](https://github.com/simstack/molecular_qm_models),
  [`simstack`](https://github.com/simstack/simstack) (`fix-git-pull`).

## Versioning

The package version is derived from Git tags ([Semantic Versioning](https://semver.org/)) by `hatch-vcs`. Do not set `project.version` by hand.

Pushes to the default branch run [python-semantic-release](https://github.com/python-semantic-release/python-semantic-release). It reads [Conventional Commits](https://www.conventionalcommits.org/) since the last tag and, when a bump is required, writes `CHANGELOG.md`, tags `vX.Y.Z`, and creates a GitHub Release.

| Prefix | Bump |
|---|---|
| `fix:` | patch |
| `feat:` | minor |
| `BREAKING CHANGE:` / `feat!:` | major |

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
