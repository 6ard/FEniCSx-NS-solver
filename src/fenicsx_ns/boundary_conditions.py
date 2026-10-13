# fenicsx_ns/boundary_conditions.py

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from dolfinx import fem
from mpi4py import MPI


class BoundaryCondition(ABC):
    @abstractmethod
    def update(self, t):
        pass

class Strong(BoundaryCondition):
    pass

class Weak(BoundaryCondition):
    @abstractmethod
    def bilinear_form(self, V, t):
        pass
    @abstractmethod
    def linear_form(self, V, t):
        pass

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

    def update(self, t):
        self.g.interpolate(self.p_of_t(t))