# fenicsx_ns/boundary_conditions.py

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from dolfinx import fem


class BoundaryCondition(ABC):
    def update(self, t):
        self.g.interpolate(self.u_of_t(t))

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

class PressurePin(Strong):
    def __init__(self, Q, point, p_of_t):
        self.g, self.p_of_t = fem.Function(Q), p_of_t
        dofs = np.array([0], dtype=np.int32)
        self.bc = fem.dirichletbc(self.g, dofs)

    def update(self, t):
        self.g.interpolate(self.p_of_t(t))