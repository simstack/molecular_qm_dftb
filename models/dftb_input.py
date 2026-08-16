from enum import Enum
from typing import List, Optional

from odmantic import Field, Model, Reference
from pydantic import model_validator

from molecular_qm_models.molecule import Molecule
from simstack.models import simstack_model
from simstack.models.files import FileStack
from simstack.util.cleaned_json_schema import cleaned_json_schema
from simstack.util.generate_ui_schema import generate_ui_schema


class DftbHamiltonian(str, Enum):
    XTB = "xTB"
    DFTB = "DFTB"


class XtbMethod(str, Enum):
    GFN1 = "GFN1-xTB"
    GFN2 = "GFN2-xTB"
    IPEA1 = "IPEA1-xTB"


class SkfSet(str, Enum):
    THREE_OB = "3ob-3-1"
    MIO = "mio-1-1"
    CUSTOM = "custom"


@simstack_model
class DftbInput(Model):
    """Input for the DFTB+ Python API calculator."""

    field_name: str = "DftbInput"
    molecule: Molecule = Reference()
    charge: int = Field(0, json_schema_extra={"description": "Net charge"})
    multiplicity: int = Field(1, json_schema_extra={"description": "Spin multiplicity"})

    hamiltonian: DftbHamiltonian = Field(
        DftbHamiltonian.XTB,
        json_schema_extra={"description": "xTB (tblite) or DFTB with Slater-Koster files"},
    )
    xtb_method: XtbMethod = Field(
        XtbMethod.GFN2, json_schema_extra={"description": "tblite xTB parametrization"}
    )
    skf_set: SkfSet = Field(
        SkfSet.THREE_OB, json_schema_extra={"description": "Slater-Koster parameter set"}
    )
    skf_prefix: str = Field(
        "",
        json_schema_extra={
            "description": "Type2FileNames Prefix (empty = DFTBPLUS_PARAM_DIR / skf_set)"
        },
    )

    scc: bool = Field(True, json_schema_extra={"description": "Self-consistent charges (DFTB)"})
    third_order: bool = Field(
        True, json_schema_extra={"description": "DFTB3 third-order + 3ob Hubbard derivs"}
    )
    max_scc_iterations: int = Field(100, json_schema_extra={"description": "Maximum SCC iterations"})
    scc_tolerance: float = Field(1.0e-5, json_schema_extra={"description": "SCC tolerance"})
    electronic_temperature: float = Field(
        300.0, json_schema_extra={"description": "Fermi filling temperature (K)"}
    )
    parser_version: int = Field(14, json_schema_extra={"description": "HSD ParserVersion"})

    compute_gradients: bool = Field(
        True, json_schema_extra={"description": "Call get_gradients()"}
    )
    compute_charges: bool = Field(
        True, json_schema_extra={"description": "Call get_gross_charges()"}
    )
    compute_cm5: bool = Field(False, json_schema_extra={"description": "Call get_cm5_charges()"})

    optimization: bool = Field(
        False, json_schema_extra={"description": "Steepest-descent using API gradients"}
    )
    max_optimization_steps: int = Field(
        100, json_schema_extra={"description": "Maximum geometry steps"}
    )
    force_tolerance: float = Field(
        1.0e-4, json_schema_extra={"description": "Max |grad| in Hartree/Bohr"}
    )

    use_hsd_file: bool = Field(False, json_schema_extra={"title": "Use existing dftb_in.hsd"})
    hsd_file: Optional[FileStack] = Field(
        None, json_schema_extra={"description": "Pre-built DFTB+ HSD input"}
    )

    use_periodic: bool = Field(False, json_schema_extra={"title": "Periodic lattice"})
    lattice_a: Optional[List[float]] = Field(
        None, json_schema_extra={"description": "Lattice vector a (Angstrom)"}
    )
    lattice_b: Optional[List[float]] = Field(
        None, json_schema_extra={"description": "Lattice vector b (Angstrom)"}
    )
    lattice_c: Optional[List[float]] = Field(
        None, json_schema_extra={"description": "Lattice vector c (Angstrom)"}
    )
    kpoint_mesh: int = Field(1, json_schema_extra={"description": "Gamma-centered NxNxN folding"})

    use_external_potential: bool = Field(
        False, json_schema_extra={"title": "Population-independent external potential"}
    )
    external_potential: Optional[List[float]] = Field(
        None, json_schema_extra={"description": "extpot[natom] in atomic units"}
    )
    external_potential_gradient: Optional[List[float]] = Field(
        None, json_schema_extra={"description": "extpotgrad[natom*3] in atomic units"}
    )

    tolerate_failure: bool = Field(
        False, json_schema_extra={"description": "Succeed even if DFTB+ raises"}
    )

    @model_validator(mode="before")
    @classmethod
    def sync_optional_toggles(cls, data):
        if not isinstance(data, dict):
            return data
        if "field_name" not in data:
            data["field_name"] = cls.__name__
        if "use_hsd_file" not in data:
            data["use_hsd_file"] = data.get("hsd_file") is not None
        if not data.get("use_hsd_file"):
            data["hsd_file"] = None
        if "use_periodic" not in data:
            data["use_periodic"] = data.get("lattice_a") is not None
        if not data.get("use_periodic"):
            data["lattice_a"] = None
            data["lattice_b"] = None
            data["lattice_c"] = None
        if "use_external_potential" not in data:
            data["use_external_potential"] = data.get("external_potential") is not None
        if not data.get("use_external_potential"):
            data["external_potential"] = None
            data["external_potential_gradient"] = None
        return data

    @model_validator(mode="after")
    def validate_options(self):
        if self.use_periodic:
            for name, vec in (
                ("lattice_a", self.lattice_a),
                ("lattice_b", self.lattice_b),
                ("lattice_c", self.lattice_c),
            ):
                if vec is None or len(vec) != 3:
                    raise ValueError(f"{name} must be a length-3 vector for periodic jobs")
        if self.use_hsd_file and self.hsd_file is None:
            raise ValueError("hsd_file is required when use_hsd_file is True")
        if self.use_external_potential:
            if not self.external_potential:
                raise ValueError("external_potential is required when the toggle is on")
            natom = len(self.molecule.atoms) if self.molecule and self.molecule.atoms else None
            if natom is not None and len(self.external_potential) != natom:
                raise ValueError("external_potential length must match the number of atoms")
            if self.external_potential_gradient is not None and natom is not None:
                if len(self.external_potential_gradient) != natom * 3:
                    raise ValueError("external_potential_gradient must have length 3 * natom")
        if self.optimization:
            self.compute_gradients = True
        return self

    @classmethod
    def json_schema(cls, recursive=True):
        schema = cleaned_json_schema(cls)
        schema["title"] = cls.__name__
        props = schema["properties"]

        xtb_method = props.pop("xtb_method", None)
        skf_set = props.pop("skf_set", None)
        skf_prefix = props.pop("skf_prefix", None)
        scc = props.pop("scc", None)
        third_order = props.pop("third_order", None)

        hsd_file = props.pop("hsd_file", None)
        lattice_a = props.pop("lattice_a", None)
        lattice_b = props.pop("lattice_b", None)
        lattice_c = props.pop("lattice_c", None)
        kpoint_mesh = props.pop("kpoint_mesh", None)
        extpot = props.pop("external_potential", None)
        extpotgrad = props.pop("external_potential_gradient", None)
        max_opt = props.pop("max_optimization_steps", None)
        force_tol = props.pop("force_tolerance", None)

        schema.setdefault("dependencies", {})
        schema["dependencies"]["hamiltonian"] = {
            "oneOf": [
                {
                    "properties": {
                        "hamiltonian": {"const": DftbHamiltonian.XTB.value},
                        "xtb_method": xtb_method,
                    }
                },
                {
                    "properties": {
                        "hamiltonian": {"const": DftbHamiltonian.DFTB.value},
                        "skf_set": skf_set,
                        "skf_prefix": skf_prefix,
                        "scc": scc,
                        "third_order": third_order,
                    }
                },
            ]
        }
        schema["dependencies"]["use_hsd_file"] = {
            "oneOf": [
                {"properties": {"use_hsd_file": {"const": False}}},
                {
                    "properties": {
                        "use_hsd_file": {"const": True},
                        "hsd_file": hsd_file,
                    }
                },
            ]
        }
        schema["dependencies"]["use_periodic"] = {
            "oneOf": [
                {"properties": {"use_periodic": {"const": False}}},
                {
                    "properties": {
                        "use_periodic": {"const": True},
                        "lattice_a": lattice_a,
                        "lattice_b": lattice_b,
                        "lattice_c": lattice_c,
                        "kpoint_mesh": kpoint_mesh,
                    }
                },
            ]
        }
        schema["dependencies"]["use_external_potential"] = {
            "oneOf": [
                {"properties": {"use_external_potential": {"const": False}}},
                {
                    "properties": {
                        "use_external_potential": {"const": True},
                        "external_potential": extpot,
                        "external_potential_gradient": extpotgrad,
                    }
                },
            ]
        }
        schema["dependencies"]["optimization"] = {
            "oneOf": [
                {"properties": {"optimization": {"const": False}}},
                {
                    "properties": {
                        "optimization": {"const": True},
                        "max_optimization_steps": max_opt,
                        "force_tolerance": force_tol,
                    }
                },
            ]
        }
        return schema

    @classmethod
    def ui_schema(cls):
        ui = generate_ui_schema(cls)
        ui["field_name"] = {"ui:widget": "hidden"}
        ui["use_hsd_file"] = {
            "ui:widget": "checkbox",
            "ui:title": "Use existing dftb_in.hsd",
        }
        ui.setdefault("hsd_file", {})["ui:condition"] = {"use_hsd_file": True}
        ui["use_periodic"] = {"ui:widget": "checkbox", "ui:title": "Periodic lattice"}
        for name in ("lattice_a", "lattice_b", "lattice_c", "kpoint_mesh"):
            ui.setdefault(name, {})["ui:condition"] = {"use_periodic": True}
        ui["use_external_potential"] = {
            "ui:widget": "checkbox",
            "ui:title": "Population-independent external potential",
        }
        ui.setdefault("external_potential", {})["ui:condition"] = {
            "use_external_potential": True
        }
        ui.setdefault("external_potential_gradient", {})["ui:condition"] = {
            "use_external_potential": True
        }
        ui["optimization"] = {"ui:widget": "checkbox", "ui:title": "Optimize geometry"}
        ui.setdefault("max_optimization_steps", {})["ui:condition"] = {"optimization": True}
        ui.setdefault("force_tolerance", {})["ui:condition"] = {"optimization": True}
        ui.setdefault("xtb_method", {})["ui:condition"] = {"hamiltonian": DftbHamiltonian.XTB.value}
        for name in ("skf_set", "skf_prefix", "scc", "third_order"):
            ui.setdefault(name, {})["ui:condition"] = {"hamiltonian": DftbHamiltonian.DFTB.value}
        return ui
