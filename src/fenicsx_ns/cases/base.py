# fenicsx_ns/cases/base.py

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import ufl


class Case(ABC):
    def __init__(self, fluid_params):
        self.fluid_params = fluid_params

    @abstractmethod
    def create_mesh(self):  pass           # returns (mesh, facet_tags)

    @abstractmethod
    def boundary_conditions(self, V, Q, facet_tags): pass

    def body_force(self, x, t):  return None

    def initial_condition(self, t):
        return None

class ExactSolution(ABC):
    def __init__(self, fluid_params):
        self.nu, self.rho = fluid_params.nu, fluid_params.rho

    @abstractmethod
    def _u(self, x, t, m) : pass

    @abstractmethod
    def _p(self, x, t, m) : pass


    def u_np(self, t) : return lambda x : np.vstack(self._u(x, t, np))
    def p_np(self, t) : return lambda x : self._p(x, t, np)
    def u_ufl(self, x, t) : return ufl.as_vector(self._u(x, t, ufl))
    def p_ufl(self, x, t) : return self._p(x, t, ufl)
