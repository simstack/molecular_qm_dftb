# Build from this capability repository:
#   docker build --build-arg UV_GIT_SHAS=$(python resolve_uv_git_shas.py pyproject.docker) --build-arg PACKAGE_VERSION=$(hatch version) -t molecular-qm-dftb:latest .
# From simstack-model:
#   docker build --build-arg UV_GIT_SHAS=$(python scripts/resolve_uv_git_shas.py molecular_qm_dftb/pyproject.docker) --build-arg PACKAGE_VERSION=$(hatch version) -t molecular-qm-dftb:latest -f molecular_qm_dftb/Dockerfile molecular_qm_dftb
# Do not pass SIMSTACK_SHA: the Dockerfile cache key is UV_GIT_SHAS.
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
    cmake \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN micromamba install -y -n base -c conda-forge setuptools && \
    micromamba clean --all --yes

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/opt/conda/bin:/root/.local/bin:$PATH"

WORKDIR /app

# Install Python runtime deps (dftbplus is built from source below).
RUN micromamba install -y -n base -c conda-forge \
    python=3.12.12 \
    numpy \
    && micromamba clean --all --yes

RUN git clone https://github.com/dftbplus/dftbplus.git /tmp/dftbplus \
 && cd /tmp/dftbplus \
 && git checkout 25.1 \
 && git submodule update --init --recursive \
 && FC=gfortran CC=gcc cmake -S . -B _build_instance \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/root/opt/dftbplus-25.1-instance \
    -DWITH_API=ON \
    -DWITH_PYTHON=ON \
    -DINSTANCE_SAFE_BUILD=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DENABLE_DYNAMIC_LOADING=ON \
    -DWITH_MPI=OFF \
    -DWITH_OMP=ON \
    -DWITH_ARPACK=OFF \
    -DWITH_POISSON=OFF \
    -DWITH_TRANSPORT=OFF \
    -DWITH_CHIMES=OFF \
 && cmake --build _build_instance -j \
 && cmake --install _build_instance \
 && test -e /root/opt/dftbplus-25.1-instance/lib/libdftbplus.so \
 && test -x /root/opt/dftbplus-25.1-instance/bin/dftb+ \
 && rm -rf /tmp/dftbplus

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
ENV DFTBPLUS_LIB=/root/opt/dftbplus-25.1-instance/lib/libdftbplus
ENV PATH="/root/opt/dftbplus-25.1-instance/bin:/opt/conda/bin:/root/.local/bin:$PATH"
ENV LD_LIBRARY_PATH=/root/opt/dftbplus-25.1-instance/lib
# Make the Python API built by cmake (WITH_PYTHON=ON) importable.
ENV PYTHONPATH=/root/opt/dftbplus-25.1-instance/lib/python3.12/site-packages
# glibc 2.41+ (Docker Desktop) rejects Fortran SOs that request an executable stack.
ENV GLIBC_TUNABLES=glibc.rtld.execstack=2

ENV UV_PYTHON=/opt/conda/bin/python
ENV UV_PROJECT_ENVIRONMENT=/opt/conda
ENV PATH="/root/opt/dftbplus-25.1-instance/bin:/opt/conda/bin:/root/.local/bin:$PATH"

# Capability package only — deps install from git via pyproject.docker.
COPY . /build/molecular_qm_dftb
WORKDIR /build/molecular_qm_dftb
# uv pip install . uses pyproject.docker. UV_GIT_SHAS is only a cache key:
# resolved commits of those git sources, so this layer rebuilds when a pinned
# branch (e.g. fix-git-pull) moves. PACKAGE_VERSION is for hatch-vcs when
# .git is not in the image (SETUPTOOLS_SCM_PRETEND_VERSION).
ARG UV_GIT_SHAS=unknown
ARG PACKAGE_VERSION=0.1.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${PACKAGE_VERSION}
RUN echo "uv git sources ${UV_GIT_SHAS}" \
 && echo "package version ${PACKAGE_VERSION}" \
 && cp pyproject.docker pyproject.toml \
 && uv pip install --system . "setuptools>=80.9.0" \
 && python -c "from molecular_qm_dftb.models.dftb_input import DftbInput; \
o=DftbInput(optimization=True); \
assert o.optimization is True and o.compute_gradients is True" \
 && python -c "from dftbplus import DftbPlus; print(DftbPlus)" \
 && python -c "import simstack, molecular_qm_models, molecular_qm_dftb; \
print('dftb+', '/root/opt/dftbplus-25.1-instance/bin/dftb+'); \
print('simstack', simstack.__file__); \
print('models', molecular_qm_models.__file__); \
print('dftb', molecular_qm_dftb.__file__, molecular_qm_dftb.__version__)"

WORKDIR /app
ENTRYPOINT ["python", "-m", "simstack.core.run_node"]
