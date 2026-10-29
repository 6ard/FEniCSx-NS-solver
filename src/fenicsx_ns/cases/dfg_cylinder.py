# fenicsx_ns/cases/dfg_cylinder.py

from __future__ import annotations

import numpy as np

from fenicsx_ns.boundary_conditions import DirichletVelocity, DoNothing
from fenicsx_ns.cases.base import Case
from fenicsx_ns.cases.meshes import create_channel_cylinder_mesh
from fenicsx_ns.parameters import FluidParameters

INLET, WALL, OUTLET, CYLINDER = 1, 2, 3, 4

class DFGCylinderCase(Case):
    # Flow around a cylinder in a channel, Schäfer & Turek (1996), DFG benchmark:
    #   2D-1: steady, Re = 20 (run in time until steady)
    #   2D-2: periodic vortex shedding, Re = 100
    #   2D-3: inflow 1.5 sin(pi t / 8) on [0, 8], Re up to 100
    # https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html
    # (dfg_benchmark2_re100.html and dfg_benchmark3_re100.html for 2D-2 and 2D-3)
    length, height, centre, radius = 2.2, 0.41, (0.2, 0.2), 0.05
    pressure_points = ((0.15, 0.2), (0.25, 0.2)) # Front and back of the cylinder, dp = p(front) - p(back)

    # Spectral reference values for 2D-1. 2D-2 and 2D-3 are only published as mesh/time-step convergence tables.
    REFERENCE = {"2D-1": {"C_D": 5.57953523384, "C_L": 0.010618948146, "dp": 0.11752016697}}

    def __init__(self, variant="2D-1", fluid_params=FluidParameters(nu=1e-3, rho=1.0), h_cylinder=0.005, h_outer=0.04,
                 ramp_time=1.0, outlet=DoNothing):
        super().__init__(fluid_params)
        if variant not in ("2D-1", "2D-2", "2D-3"):
            raise ValueError(f"Unknown DFG variant {variant!r}, expected '2D-1', '2D-2' or '2D-3'")

        self.variant, self.outlet = variant, outlet
        self.h_cylinder, self.h_outer, self.ramp_time = h_cylinder, h_outer, ramp_time
        self.U_max = 0.3 if variant == "2D-1" else 1.5
        self.U_mean, self.D = 2 * self.U_max / 3, 2 * self.radius # Scales of the drag and lift coefficients
        self.reference = self.REFERENCE.get(variant)

    def inflow_amplitude(self, t):
        if self.variant == "2D-3":
            return self.U_max * np.sin(np.pi * t / 8)
        # 2D-1 and 2D-2 have constant inflow; ramp it up smoothly from the fluid at rest
        if t < self.ramp_time:
            return self.U_max * 0.5 * (1 - np.cos(np.pi * t / self.ramp_time))
        return self.U_max

    def inflow(self, t):
        U = self.inflow_amplitude(t)
        return lambda x: np.vstack((4 * U * x[1] * (self.height - x[1]) / self.height**2, np.zeros_like(x[1])))

    def create_mesh(self):
        markers = {"inlet": INLET, "wall": WALL, "outlet": OUTLET, "cylinder": CYLINDER}
        return create_channel_cylinder_mesh(self.length, self.height, self.centre, self.radius,
                                            self.h_cylinder, self.h_outer, markers)

    def boundary_conditions(self, V, Q, facet_tags):
        no_slip = lambda t: lambda x: np.zeros((2, x.shape[1]))
        return [DirichletVelocity(V, facet_tags, marker=INLET, u_of_t=self.inflow),
                DirichletVelocity(V, facet_tags, marker=WALL, u_of_t=no_slip),
                DirichletVelocity(V, facet_tags, marker=CYLINDER, u_of_t=no_slip),
                self.outlet(OUTLET)]

    def initial_condition(self, t):
        return lambda x: np.zeros((2, x.shape[1]))
