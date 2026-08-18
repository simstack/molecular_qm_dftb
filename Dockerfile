# Build from this capability repository:
#   docker build -t molecular-qm-dftb:latest .
# From simstack-model:
#   docker build -t molecular-qm-dftb:latest -f molecular_qm_dftb/Dockerfile molecular_qm_dftb
#
# Dual-use: capability tree is not installable on host (no pyproject.toml).
# In the image, pyproject.docker is renamed and the package is pip-installed;
# models / simstack come from git (see pyproject.docker).
FROM mambaorg/micromamba:latest

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN micromamba install -y -n base -c conda-forge setuptools && \
    micromamba clean --all --yes

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/opt/conda/bin:/root/.local/bin:$PATH"

WORKDIR /app

# Serial (OpenMP) DFTB+ 25.1 plus the ctypes Python API. tblite is a
# runtime dep of the conda-forge dftbplus build, so GFN1/GFN2-xTB works
# without Slater-Koster files. Do not conda-install pymatgen: it pulls
# X11/matplotlib and Kaniko OOMs on the standard GitLab runner.
RUN micromamba install -y -n base -c conda-forge \
    python=3.12.12 \
    numpy \
    "dftbplus=25.1=nompi_*" \
    dftbplus-python=25.1 \
    && micromamba clean --all --yes \
    && test -e /opt/conda/lib/libdftbplus.so \
    && test -x /opt/conda/bin/dftb+

# 3ob-3-1 / mio-1-1 SKF sets (CC-BY-SA; cite the 3ob/mio README references).
RUN mkdir -p /opt/dftbplus/params \
 && curl -fsSL https://github.com/dftbparams/3ob/archive/refs/heads/main.tar.gz \
    | tar -xz -C /tmp \
 && mv /tmp/3ob-main/skfiles /opt/dftbplus/params/3ob-3-1 \
 && curl -fsSL https://github.com/dftbparams/mio/archive/refs/heads/main.tar.gz \
    | tar -xz -C /tmp \
 && mv /tmp/mio-main/skfiles /opt/dftbplus/params/mio-1-1 \
 && test -f /opt/dftbplus/params/3ob-3-1/H-H.skf \
 && test -f /opt/dftbplus/params/mio-1-1/H-H.skf \
 && rm -rf /tmp/3ob-main /tmp/mio-main

ENV DFTBPLUS_PARAM_DIR=/opt/dftbplus/params
ENV DFTBPLUS_LIB=/opt/conda/lib/libdftbplus
# glibc 2.41+ (Docker Desktop) rejects Fortran SOs that request an executable stack.
ENV GLIBC_TUNABLES=glibc.rtld.execstack=2

ENV UV_PYTHON=/opt/conda/bin/python
ENV UV_PROJECT_ENVIRONMENT=/opt/conda
ENV PATH="/opt/conda/bin:/root/.local/bin:$PATH"

# Capability package only — deps install from git via pyproject.docker.
COPY . /build/molecular_qm_dftb
WORKDIR /build/molecular_qm_dftb
RUN cp pyproject.docker pyproject.toml \
 && uv pip install --system . "setuptools>=80.9.0" \
 && python -c "import dftbplus, simstack, molecular_qm_models, molecular_qm_dftb; \
from dftbplus import DftbPlus; \
print('dftbplus', dftbplus.__file__); \
print('simstack', simstack.__file__); \
print('models', molecular_qm_models.__file__); \
print('dftb', molecular_qm_dftb.__file__)"

WORKDIR /app
ENTRYPOINT ["python", "-m", "simstack.core.run_node"]
