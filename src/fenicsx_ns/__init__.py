import dolfinx

if not dolfinx.__version__.startswith("0.11"):
    raise ImportError(f"This package requires dolfinx version 0.10.x or higher, found version: {dolfinx.__version__}")

from fenicsx_ns.boundary_conditions import  DirichletVelocity, PressurePin
from fenicsx_ns.observers import Observer, ErrorNorm
from fenicsx_ns.parameters import FluidParameters, TimeParameters, StabilizationParameters
from fenicsx_ns.time_schemes import TimeScheme

from fenicsx_ns.solvers import NavierStokesSolver

__all__ = [
    "DirichletVelocity",
    "PressurePin",
    "Observer",
    "ErrorNorm",
    "FluidParameters",
    "TimeParameters",
    "StabilizationParameters",
    "TimeScheme",
    "NavierStokesSolver"
]