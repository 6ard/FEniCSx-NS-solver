# fenicsx_ns/solvers/navier_stokes.py

from __future__ import annotations

import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

from fenicsx_ns.boundary_conditions import PressurePin, Strong, Weak
from fenicsx_ns.solvers.base import SteadyState, TimeDependence
from fenicsx_ns.utils import global_integral, constant, domain_volume, subtract_mean


def t_h_space(msh):
    gdim = msh.geometry.dim
    V = fem.functionspace(msh, ("Lagrange", 2, (gdim,))) # P2
    Q = fem.functionspace(msh, ("Lagrange", 1)) # P1
    return V, Q

def eps(z):
    return ufl.sym(ufl.grad(z))

def conv(u_ns, u, v):
    return ufl.inner(ufl.dot(ufl.grad(u), u_ns), v)

class SteadyNavierStokesSolver(SteadyState):
    pass

class NavierStokesSolver(TimeDependence):
    def __init__(self, case, time_params, scheme, **kwargs):
        super().__init__(case, time_params, scheme, **kwargs)
        self.msh, self.facet_tags = self.case.create_mesh()

        self._spaces() # Creates V, Q, u_h, u_n, u_nm, p_h
        self._constants() # Creates dx, ds, n, rho_c, mu_c, dt_c, grad_div_c, backflow_c

        self.bcs = self.case.boundary_conditions(self.V, self.Q, self.facet_tags)
        self.weak_bcs = [b for b in self.bcs if isinstance(b, Weak)]

        # Without a weak (open) boundary the pressure is only determined up to a constant
        self.pressure_up_to_constant = not self.weak_bcs
        if self.pressure_up_to_constant:
            self.bcs.append(PressurePin(self.Q))

        self._forms()
        self._create_problem()
        self._normalize()

        self._apply_initial_conditions()

    def _spaces(self) :
        self.V, self.Q = t_h_space(self.msh)
        self.u_h = fem.Function(self.V, name="Velocity")
        self.u_n = fem.Function(self.V)
        self.u_nm = fem.Function(self.V)

        self.p_h = fem.Function(self.Q, name="Pressure")

    def _constants(self) :
        self.x_c, self.t_c = ufl.SpatialCoordinate(self.msh), constant(self.msh, self.t)
        self.dx = ufl.Measure("dx", self.msh)
        self.ds = ufl.Measure("ds", self.msh, subdomain_data=self.facet_tags)
        self.n = ufl.FacetNormal(self.msh)

        self.rho_c, self.mu_c, self.dt_c, self.grad_div_c, self.backflow_c = [constant(self.msh, v) for v in (self.case.fluid_params.rho,
                                                                              self.case.fluid_params.mu, self.dt, self.stabilization.grad_div,
                                                                              self.stabilization.backflow)]

        first = self.scheme.startup or self.scheme
        
        self.a_c = [constant(self.msh, c) for c in first.a]
        self.e_c = [constant(self.msh, c) for c in first.e]    


    def _forms(self) :
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        p, q = ufl.TrialFunction(self.Q), ufl.TestFunction(self.Q)

        a0, a1, a2 = self.a_c
        e0, e1 = self.e_c

        u_ns = e0 * self.u_n + e1 * self.u_nm

        a00 = (self.rho_c *a0 / self.dt_c * ufl.inner(u, v) * self.dx
                + 0.5 * self.rho_c * (conv(u_ns, u, v) - conv(u_ns, v, u)) * self.dx
                + 2 * self.mu_c * ufl.inner(eps(u), eps(v)) * self.dx
                + self.grad_div_c * ufl.div(u) * ufl.div(v) * self.dx)

        a01 = - p * ufl.div(v) * self.dx
        a10 = - q * ufl.div(u) * self.dx

        L0 = -self.rho_c / self.dt_c * ufl.inner(a1 * self.u_n + a2 * self.u_nm, v) * self.dx
        L1 = ufl.ZeroBaseForm((q,))

        for b in self.weak_bcs:
            ds, u_ns_n = self.ds(b.marker), ufl.dot(u_ns, self.n)
            a00 += b.bilinear_form(u, v, self.n, self.ds, self.mu_c)
            # The skew-symmetric convection leaves -rho/2 (u_ns.n) u on open boundaries; add it back
            a00 += 0.5 * self.rho_c * u_ns_n * ufl.inner(u, v) * ds
            if self.stabilization.backflow > 0:
                # Backflow stabilization: removes the energy convection carries in where u_ns.n < 0
                a00 -= 0.5 * self.backflow_c * self.rho_c * ufl.min_value(u_ns_n, 0) * ufl.inner(u, v) * ds
            L0 += b.linear_form(v, self.n, self.ds, self.x_c, self.t_c, self.mu_c)

        zero = constant(self.msh, 0.0)
        a11 = zero * p * q * self.dx

        self.a = fem.form([[a00, a01], [a10, a11]])
        self.L = fem.form([L0, L1])
        
    def _normalize(self):
        self._vol = domain_volume(self.msh)
        self._p_int = fem.form(self.p_h * self.dx)


    def _create_problem(self) :
        self.problem = LinearProblem(
            self.a, self.L, u=[self.u_h, self.p_h], bcs=[b.bc for b in self.bcs if isinstance(b, Strong)],
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

        if self.pressure_up_to_constant:
            subtract_mean(self.p_h, self._p_int, self._vol)

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
