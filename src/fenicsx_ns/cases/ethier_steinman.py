# fenicsx_ns/cases/ethier_steinman.py

from __future__ import annotations

import numpy as np
from dolfinx import mesh
from mpi4py import MPI

from fenicsx_ns.boundary_conditions import DirichletVelocity, DoNothing
from fenicsx_ns.cases.base import Case, ExactSolution
from fenicsx_ns.cases.meshes import create_box_mesh, tag_exterior_facets

WALL, OUTLET = 1, 2

class EthierSteinmanExact(ExactSolution):
    def __init__(self, fluid_params, a, d):
        super().__init__(fluid_params)
        self.a, self.d = a, d

    def _F(self, t, m) :
        return m.exp(-self.nu*self.d**2*t)

    def _u(self, x, t, m) :
        return ( - self.a *(m.exp(self.a*x[0])*m.sin(self.a*x[1] + self.d*x[2]) + m.exp(self.a*x[2])*m.cos(self.a*x[0] + self.d*x[1])) * self._F(t, m),
                             - self.a *(m.exp(self.a*x[1])*m.sin(self.a*x[2] + self.d*x[0]) + m.exp(self.a*x[0])*m.cos(self.a*x[1] + self.d*x[2])) * self._F(t, m),
                             - self.a *(m.exp(self.a*x[2])*m.sin(self.a*x[0] + self.d*x[1]) + m.exp(self.a*x[1])*m.cos(self.a*x[2] + self.d*x[0])) * self._F(t, m) ) 

    def _p(self, x, t, m) :
        return - self.rho * self.a**2 / 2 * ( m.exp(2*self.a*x[0]) + m.exp(2*self.a*x[1]) + m.exp(2*self.a*x[2])
                                  + 2 * m.sin(self.a*x[0] + self.d*x[1]) * m.cos(self.a*x[2] + self.d*x[0]) * m.exp(self.a*(x[1] + x[2]))
                                  + 2 * m.sin(self.a*x[1] + self.d*x[2]) * m.cos(self.a*x[0] + self.d*x[1]) * m.exp(self.a*(x[0] + x[2]))
                                  + 2 * m.sin(self.a*x[2] + self.d*x[0]) * m.cos(self.a*x[1] + self.d*x[2]) * m.exp(self.a*(x[0] + x[1])) ) * self._F(t, m)**2


class EthierSteinmanCase(Case):
    # outlet: Weak condition class (DoNothing or TrueDoNothing) on x = L, None for Dirichlet on the whole boundary
    def __init__(self, N, fluid_params, a, d, outlet=DoNothing):
        super().__init__(fluid_params)
        self.N, self.L, self.outlet = N, 1, outlet
        self.exact = EthierSteinmanExact(self.fluid_params, a, d)

    def create_mesh(self):
        msh = create_box_mesh((-self.L,) * 3, (self.L,) * 3, (self.N,) * 3, mesh.CellType.hexahedron)
        return msh, tag_exterior_facets(msh, WALL, {OUTLET: lambda x: np.isclose(x[0], self.L)} if self.outlet else None)

    def boundary_conditions(self, V, Q, facet_tags):
        bcs = [DirichletVelocity(V, facet_tags, marker=WALL, u_of_t=self.exact.u_np)]
        if self.outlet:
            bcs.append(self.outlet(OUTLET, exact=self.exact))
        return bcs

    def initial_condition(self, t):
        return self.exact.u_np(t)