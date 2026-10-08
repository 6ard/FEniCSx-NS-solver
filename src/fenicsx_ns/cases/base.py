# fenicsx_ns/cases/base.py

from __future__ import annotations
from abc import ABC, abstractmethod

class Case(ABC):
    def __init__(self, fluid_params):
        self.fluid_params = fluid_params

    @abstractmethod
    def create_mesh(self):  pass           # returns (mesh, facet_tags)

    @abstractmethod
    def boundary_conditions(self, V, Q, facet_tags): pass

    def body_force(self, x, t):  return None

    @abstractmethod
    def initial_condition(self, t): pass    # callable for interpolation
    pressure_nullspace: bool = False