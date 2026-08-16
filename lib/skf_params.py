"""3ob-3-1 and mio-1-1 Slater-Koster metadata for DFTB+ HSD generation."""

from __future__ import annotations

import os
from pathlib import Path

# Hubbard derivatives (atomic units) from the 3ob-3-1 README.
THREE_OB_HUBBARD_DERIVS = {
    "Br": -0.0573,
    "C": -0.1492,
    "Ca": -0.0340,
    "Cl": -0.0697,
    "F": -0.1623,
    "H": -0.1857,
    "I": -0.0433,
    "K": -0.0339,
    "Mg": -0.02,
    "N": -0.1535,
    "Na": -0.0454,
    "O": -0.1575,
    "P": -0.14,
    "S": -0.11,
    "Zn": -0.03,
}

THREE_OB_MAX_ANGULAR_MOMENTUM = {
    "Br": "d",
    "C": "p",
    "Ca": "p",
    "Cl": "d",
    "F": "p",
    "H": "s",
    "I": "d",
    "K": "p",
    "Mg": "p",
    "N": "p",
    "Na": "p",
    "O": "p",
    "P": "d",
    "S": "d",
    "Zn": "d",
}

# mio-1-1: organic ONCH + S/P. No third-order Hubbard derivatives in the set.
MIO_MAX_ANGULAR_MOMENTUM = {
    "C": "p",
    "H": "s",
    "N": "p",
    "O": "p",
    "P": "d",
    "S": "d",
}

THREE_OB_DAMP_XH_EXPONENT = 4.00

DEFAULT_PARAM_DIR = "/opt/dftbplus/params"


def param_root() -> Path:
    return Path(os.environ.get("DFTBPLUS_PARAM_DIR", DEFAULT_PARAM_DIR))


def skf_prefix_for(skf_set: str, custom_prefix: str = "") -> str:
    """Return a Type2FileNames Prefix that DFTB+ can open.

    The trailing slash is required so files resolve as ``{prefix}{A}-{B}.skf``.
    """
    if custom_prefix:
        prefix = custom_prefix
    else:
        prefix = str(param_root() / skf_set)
    if not prefix.endswith(("/", "\\")):
        prefix = prefix + "/"
    return prefix.replace("\\", "/")
