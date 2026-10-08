# fenicsx_ns/boundary_conditions.py

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from dolfinx import fem, mesh


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
    def __init__(self, Q, point, p_of_t):
        self.g, self.p_of_t = fem.Function(Q), p_of_t
        dofs = fem.locate_dofs_geometrical(Q, lambda x: np.isclose(x[0], point[0]) & np.isclose(x[1], point[1]))
        self.bc = fem.dirichletbc(self.g, dofs)

    def update(self, t):
        self.g.interpolate(self.p_of_t(t))