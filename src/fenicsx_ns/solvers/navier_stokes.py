# fenicsx_ns/solvers/navier_stokes.py

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem.petsc import LinearProblem

from fenicsx_ns.boundary_conditions import  PressurePin, DirichletVelocity
from fenicsx_ns.solvers.base import SteadyState, TimeDependence

if TYPE_CHECKING:
    from fenicsx_ns.cases.base import Case
    from fenicsx_ns.parameters import TimeParameters
    from fenicsx_ns.time_schemes import TimeScheme

def t_h_space(msh):
    gdim = msh.geometry.dim
    V = fem.functionspace(msh, ("Lagrange", 2, (gdim,))) # P2
    Q = fem.functionspace(msh, ("Lagrange", 1)) # P1
    return V, Q

class SteadyNavierStokesSolver(SteadyState):
    pass

class NavierStokesSolver(TimeDependence):
    def __init__(self, case, time_params, scheme, **kwargs):
        super().__init__(case, time_params, scheme, **kwargs)
        self.msh, self.facet_tags = self.case.create_mesh()

        self._spaces() # Creates V, Q, u_h, u_n, u_nm, p_h
        self._constants() # Creates rho_c, mu_c, dt_c, grad_div_c

        self.bcs = self.case.boundary_conditions(self.V, self.Q, self.facet_tags)


        self._forms()
        self._create_problem()

        self._apply_initial_conditions()

    def _spaces(self) :
        self.V, self.Q = t_h_space(self.msh)
        self.u_h = fem.Function(self.V)
        self.u_n = fem.Function(self.V)
        self.u_nm = fem.Function(self.V)

        self.p_h = fem.Function(self.Q)

    def _fconst(self, v):
        return fem.Constant(self.msh, PETSc.ScalarType(v))

    def _constants(self) :
        self.x_c, self.t_c = ufl.SpatialCoordinate(self.msh), self._fconst(self.t)
        self.dx = ufl.Measure("dx", self.msh)

        self.rho_c, self.mu_c, self.dt_c, self.grad_div_c = [self._fconst(v) for v in (self.case.fluid_params.rho, 
                                                                              self.case.fluid_params.mu, self.dt, self.stabilization.grad_div)]

        first = self.scheme.startup or self.scheme
        
        self.a_c = [self._fconst(c) for c in first.a]
        self.e_c = [self._fconst(c) for c in first.e]    

    def _split_bcs(self, V, Q, facet_tags):
        pass

    def _forms(self, ) :
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        p, q = ufl.TrialFunction(self.Q), ufl.TestFunction(self.Q)

        a0, a1, a2 = self.a_c
        e0, e1 = self.e_c

        u_ns = e0 * self.u_n + e1 * self.u_nm
        eps = lambda z: ufl.sym(ufl.grad(z))
        conv = lambda u_ns, u, v: ufl.inner(ufl.dot(ufl.grad(u), u_ns), v)

        a00 = (self.rho_c *a0 / self.dt_c * ufl.inner(u, v) * self.dx
                + 0.5 * self.rho_c * (conv(u_ns, u, v) - conv(u_ns, v, u)) * self.dx
                + 2 * self.mu_c * ufl.inner(eps(u), eps(v)) * self.dx
                + self.grad_div_c * ufl.div(u) * ufl.div(v) * self.dx)

        a01 = - p * ufl.div(v) * self.dx
        a10 = - q * ufl.div(u) * self.dx

        L0 = -self.rho_c / self.dt_c * ufl.inner(a1 * self.u_n + a2 * self.u_nm, v) * self.dx
        L1 = ufl.ZeroBaseForm((q,))

        zero = self._fconst(0.0)
        a11 = zero * p * q * self.dx

        self.a = fem.form([[a00, a01], [a10, a11]])
        self.L = fem.form([L0, L1])
        
        one = self._fconst(1.0)
        self._p_int = fem.form(self.p_h * self.dx)
        self._vol = self.msh.comm.allreduce(fem.assemble_scalar(fem.form(one * self.dx)), op=MPI.SUM)

    def _create_problem(self) :
        self.problem = LinearProblem(
            self.a, self.L, u=[self.u_h, self.p_h], bcs=[b.bc for b in self.bcs],
            petsc_options_prefix="ns_",
            petsc_options=self.petsc_options,
        )

    def _apply_initial_conditions(self):
        self.u_n.interpolate(self.case.initial_condition(self.t0))
        if self.scheme.order == 2 and self.scheme.startup is None:
            self.u_nm.interpolate(self.case.initial_condition(self.t0 - self.dt))
        self.u_h.x.array[:] = self.u_n.x.array # Now u_h represent the state at t0

    def update_boundary_data(self, t):
        self.t_c.value = t
        for b in self.bcs:
            b.update(t)

    def solve(self):
        self.problem.solve()
        p_mean = self.msh.comm.allreduce(fem.assemble_scalar(self._p_int), op=MPI.SUM) / self._vol
        self.p_h.x.array[:] -= p_mean

    def advance_history(self):
        self.u_nm.x.array[:] = self.u_n.x.array
        self.u_n.x.array[:] = self.u_h.x.array
        self.u_nm.x.scatter_forward(); self.u_n.x.scatter_forward()

        if self.scheme.startup is not None and self.step == 1:
            self._set_scheme(self.scheme)

    def _set_scheme(self, scheme):
        for c, val in zip(self.a_c, scheme.a):
            c.value = val

        for c, val in zip(self.e_c, scheme.e):
            c.value = val
