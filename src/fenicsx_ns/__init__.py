import dolfinx

if not dolfinx.__version__.startswith("0.11"):
    raise ImportError(f"This package requires dolfinx version 0.11 or higher, found version: {dolfinx.__version__}")

from fenicsx_ns.boundary_conditions import DirichletVelocity, PressurePin
from fenicsx_ns.observers import ErrorNorm, Observer, VTXOutput, Progress
from fenicsx_ns.utils import global_integral, constant
from fenicsx_ns.cases.meshes import create_box_mesh, tag_exterior_facets
from fenicsx_ns.verification import spatial_study, temporal_study
from fenicsx_ns.parameters import (
    FluidParameters,
    StabilizationParameters,
    TimeParameters,
)
from fenicsx_ns.solvers import NavierStokesSolver
from fenicsx_ns.time_schemes import TimeScheme

__all__ = [
    "DirichletVelocity",
    "PressurePin",
    "Observer",
    "ErrorNorm",
    "Progress",
    "VTXOutput",
    "global_integral",
    "constant",
    "create_box_mesh",
    "tag_exterior_facets",
    "spatial_study",
    "temporal_study",
    "FluidParameters",
    "TimeParameters",
    "StabilizationParameters",
    "TimeScheme",
    "NavierStokesSolver"
]