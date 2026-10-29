# fenicsx_ns/boundary_conditions.py

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI


class BoundaryCondition(ABC):
    def update(self, t):
        pass

class Strong(BoundaryCondition):
    pass

class Weak(BoundaryCondition):
    # Natural condition (S(u) - p I) n = h on the facets with this marker, S(u) being the subclass's viscous stress.
    # h = 0 unless exact (an ExactSolution) is given, then h is taken from it so the exact solution satisfies the condition.
    def __init__(self, marker, exact=None):
        self.marker, self.exact = marker, exact

    @abstractmethod
    def viscous_stress(self, u, mu):
        pass

    def bilinear_form(self, u, v, n, ds, mu):
        return 0

    def linear_form(self, v, n, ds, x, t, mu):
        if self.exact is None:
            return 0
        u_ex, p_ex = self.exact.u_ufl(x, t), self.exact.p_ufl(x, t)
        h = ufl.dot(self.viscous_stress(u_ex, mu) - p_ex * ufl.Identity(len(u_ex)), n)
        return ufl.inner(h, v) * ds(self.marker)

class TrueDoNothing(Weak):
    # Traction-free: (2 mu eps(u) - p I) n = h. This is the natural condition of the solver's
    # 2 mu eps(u) : eps(v) term, so nothing is added to the bilinear form.
    def viscous_stress(self, u, mu):
        return 2 * mu * ufl.sym(ufl.grad(u))

class DoNothing(Weak):
    # Classic do-nothing (Heywood, Rannacher & Turek 1996): (mu grad(u) - p I) n = h.
    # Unlike TrueDoNothing it lets Poiseuille flow leave the domain undisturbed.
    def viscous_stress(self, u, mu):
        return mu * ufl.grad(u)

    def bilinear_form(self, u, v, n, ds, mu):
        # 2 mu eps(u) : eps(v) also leaves mu grad(u)^T n on the boundary; remove it
        return -mu * ufl.inner(ufl.dot(ufl.grad(u).T, n), v) * ds(self.marker)


class DirichletVelocity(Strong):
    def __init__(self, V, facet_tags, marker, u_of_t):
        self.g, self.u_of_t = fem.Function(V), u_of_t
        fdim = V.mesh.topology.dim - 1
        dofs = fem.locate_dofs_topological(V, fdim, facet_tags.find(marker))
        self.bc = fem.dirichletbc(self.g, dofs)

    def update(self, t):
        self.g.interpolate(self.u_of_t(t))

class PressurePin(Strong):
    def __init__(self, Q) : 
        comm = Q.mesh.comm
        owns_dof = comm.rank == 0 and Q.dofmap.index_map.size_local > 0

        dofs = np.array([0] if owns_dof else [], dtype=np.int32)

        if comm.allreduce(dofs.size, op=MPI.SUM) != 1:
            raise ValueError("Pressure pinning requires exactly one global dof.")

        self.bc = fem.dirichletbc(fem.Constant(Q.mesh, 0.0), dofs, Q)
