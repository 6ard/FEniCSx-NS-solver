# fenicsx_ns/cases/taylor_green.py

from __future__ import annotations



import ufl 
import numpy as np
from fenicsx_ns.cases.base import Case
from fenicsx_ns.boundary_conditions import DirichletVelocity, PressurePin

from dolfinx import fem, mesh
from mpi4py import MPI


def make_TG_mesh(N, L):
    msh = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([L, L])], [N, N], mesh.CellType.quadrilateral)
    fdim = msh.topology.dim - 1

    # Tags boundary facet with marker 1, using locate_entities_boundary and meshtags
    def boundary(x):
        return np.isclose(x[0], 0) | np.isclose(x[0], L) | np.isclose(x[1], 0) | np.isclose(x[1], L)
    
    facet_indices = mesh.locate_entities_boundary(msh, fdim, boundary)
    facet_tags = mesh.meshtags(msh, fdim, facet_indices, 1)

    return msh, facet_tags

class TaylorGreenExact:
    def __init__(self, fluid_params, U0, wave_number):
        self.U0, self.nu, self.k, self.rho = U0, fluid_params.nu, wave_number, fluid_params.rho

    def _F(self, t, m) :
        return m.exp(-2*self.nu*self.k**2*t)

    def _u(self, x, t, m) :
        F = self._F(t, m)
        return (self.U0 * m.sin(self.k*x[0]) * m.cos(self.k*x[1]) * F, 
                            -self.U0 * m.cos(self.k*x[0]) * m.sin(self.k*x[1]) * F)

    def _p(self, x, t, m) :
        F = self._F(t, m)
        return self.rho * self.U0**2 / 4 * (m.cos(2*self.k*x[0]) + m.cos(2*self.k*x[1])) * F**2

    def u_np(self, t) : return lambda x : np.vstack(self._u(x, t, np))
    def p_np(self, t) : return lambda x : self._p(x, t, np)
    def u_ufl(self, x, t) : return ufl.as_vector(self._u(x, t, ufl))
    def p_ufl(self, x, t) : return self._p(x, t, ufl)

class TaylorGreenCase(Case):
    pressure_nullspace = False

    def __init__(self, N, fluid_params, U0, wave_number):
        super().__init__(fluid_params)
        self.N, self.L = N, 2 * np.pi / wave_number
        self.fluid_params = fluid_params
        self.exact = TaylorGreenExact(self.fluid_params, U0, wave_number)

    def create_mesh(self):
        return make_TG_mesh(self.N, self.L) 

    def boundary_conditions(self, V, Q, facet_tags):
        return [DirichletVelocity(V, facet_tags, marker=1, u_of_t=self.exact.u_np), PressurePin(Q, point=(0, 0), p_of_t=self.exact.p_np)]

    def initial_condition(self, t):
        return self.exact.u_np(t)