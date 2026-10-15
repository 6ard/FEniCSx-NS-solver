# fenicsx_ns/cases/taylor_green.py

from __future__ import annotations

import numpy as np
from dolfinx import mesh
from mpi4py import MPI

from fenicsx_ns.boundary_conditions import DirichletVelocity
from fenicsx_ns.cases.base import Case, ExactSolution
from fenicsx_ns.cases.meshes import create_box_mesh, tag_exterior_facets

WALL = 1

class TaylorGreenExact(ExactSolution):
    def __init__(self, fluid_params, U0, wave_number):
        super().__init__(fluid_params)
        self.U0, self.k = U0, wave_number

    def _F(self, t, m) :
        return m.exp(-2*self.nu*self.k**2*t)

    def _u(self, x, t, m) :
        F = self._F(t, m)
        return (self.U0 * m.sin(self.k*x[0]) * m.cos(self.k*x[1]) * F, 
                            -self.U0 * m.cos(self.k*x[0]) * m.sin(self.k*x[1]) * F)

    def _p(self, x, t, m) :
        F = self._F(t, m)
        return self.rho * self.U0**2 / 4 * (m.cos(2*self.k*x[0]) + m.cos(2*self.k*x[1])) * F**2

 

class TaylorGreenCase(Case):
    pressure_up_to_constant = True

    def __init__(self, N, fluid_params, U0, wave_number):
        super().__init__(fluid_params)
        self.N, self.L = N, 2 * np.pi / wave_number
        self.exact = TaylorGreenExact(self.fluid_params, U0, wave_number)

    def create_mesh(self):
        msh = create_box_mesh((0,) * 2, (self.L,) * 2, (self.N,) * 2, mesh.CellType.quadrilateral)
        return msh, tag_exterior_facets(msh, WALL) 

    def boundary_conditions(self, V, Q, facet_tags):
        return [DirichletVelocity(V, facet_tags, marker=1, u_of_t=self.exact.u_np)]

    def initial_condition(self, t):
        return self.exact.u_np(t)