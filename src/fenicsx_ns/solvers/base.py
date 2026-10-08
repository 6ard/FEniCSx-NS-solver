# fenicsx_ns/solvers/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from fenicsx_ns.parameters import StabilizationParameters


def DirectLU():
    return {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}

class Solver(ABC):
    def __init__(self, case, time_params, stabilization, petsc_options=None, observers=()):
        self.case, self.time_params, self.stabilization = case, time_params, stabilization
        if petsc_options is None:
            self.petsc_options = self.default_preconditioner()
        else : self.petsc_options = petsc_options
        self.observers = list(observers)

    def default_preconditioner(self):
        return DirectLU()

    def solve(self):
        self.ksp.solve(self.b, self.x)

class SteadyState(Solver):
    pass

class TimeDependence(Solver): 
    def __init__(self, case, time_params, scheme, *, stabilization=StabilizationParameters(), petsc_options=None, observers=()):
        super().__init__(case, time_params, stabilization, petsc_options=petsc_options, observers=observers)
        self.t0, self.dt, self.T = self.time_params.t0, self.time_params.dt, self.time_params.T
        self.t = self.t0
        
        self.scheme, self.stabilization = scheme, stabilization
        
    def run(self) :
        self.t = self.t0
        for obs in self.observers: obs.on_start(self)

        num_steps = round((self.T - self.t0) / self.dt)

        for step in range(1, num_steps + 1):
            self.step = step
            self.t = self.t0 + step * self.dt
            self.update_boundary_data(self.t)
            self.assemble_step()
            self.solve()
            for obs in self.observers: obs.on_step(self) 
            self.advance_history()

        for obs in self.observers: obs.on_end(self)

    @abstractmethod 
    def update_boundary_data(self, t):
        pass

    def assemble_step(self):
        pass

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def advance_history(self):
        pass    