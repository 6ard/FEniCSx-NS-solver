# fenicsx_ns/cases/ethier_steinman.py

from __future__ import annotations

import numpy as np
from dolfinx import mesh
from mpi4py import MPI

from fenicsx_ns.boundary_conditions import DirichletVelocity
from fenicsx_ns.cases.base import Case, ExactSolution


def make_ES_mesh(N, L):
    msh = mesh.create_box(MPI.COMM_WORLD, [np.array([-L, -L, -L]), np.array([L, L, L])], [N, N, N], mesh.CellType.hexahedron)
    fdim = msh.topology.dim - 1

    # Tags boundary facet with marker 1, using locate_entities_boundary and meshtags
    def boundary(x):
        return np.isclose(x[0], 0) | np.isclose(x[0], L) | np.isclose(x[1], 0) | np.isclose(x[1], L) | np.isclose(x[2], 0) | np.isclose(x[2], L)
    
    facet_indices = mesh.locate_entities_boundary(msh, fdim, boundary)
    facet_tags = mesh.meshtags(msh, fdim, facet_indices, 1)

    return msh, facet_tags

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
    pressure_up_to_constant = True

    def __init__(self, N, fluid_params, a, d):
        super().__init__(fluid_params)
        self.N, self.L = N, 1
        self.exact = EthierSteinmanExact(self.fluid_params, a, d)

    def create_mesh(self):
        return make_ES_mesh(self.N, self.L) 

    def boundary_conditions(self, V, Q, facet_tags):
        return [DirichletVelocity(V, facet_tags, marker=1, u_of_t=self.exact.u_np)]

    def initial_condition(self, t):
        return self.exact.u_np(t)