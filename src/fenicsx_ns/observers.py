# fenicsx_ns/observers.py
from __future__ import annotations

from abc import ABC

import numpy as np
from ufl import inner, grad
from dolfinx import fem
from mpi4py import MPI


class Observer(ABC):
    def on_start(self, solver):
        pass

    def on_step(self, solver):
        pass

    def on_end(self, solver):
        pass

class ErrorNorm(Observer): 
    def __init__(self, exact, quad_degree=8):
        self.exact, self.quad_degree = exact, quad_degree

    def on_start(self, solver):
        dx = solver.dx(degree=self.quad_degree)

        e_u, e_p = solver.u_h - self.exact.u_ufl(solver.x_c, solver.t_c), solver.p_h - self.exact.p_ufl(solver.x_c, solver.t_c)
        self.f_uL2 = fem.form(inner(e_u, e_u) * dx)
        self.f_uH1 = fem.form(inner(grad(e_u), grad(e_u)) * dx)
        self.f_pL2 = fem.form(e_p * e_p * dx)

        self.hist = {"t": [], "u_L2": [], "u_H1": [], "p_L2": []}

    def _norm(self, solver, f) :
        return np.sqrt(solver.msh.comm.allreduce(fem.assemble_scalar(f), op=MPI.SUM))

    def on_step(self, solver) :
        self.hist["t"].append(solver.t)
        self.hist["u_L2"].append(self._norm(solver, self.f_uL2))
        self.hist["u_H1"].append(self._norm(solver, self.f_uH1))
        self.hist["p_L2"].append(self._norm(solver, self.f_pL2))

    def on_end(self, solver) : 
        print("Final error norms: u_L2 = {:.3e}, u_H1 = {:.3e}, p_L2 = {:.3e}".format(
            self.hist["u_L2"][-1], self.hist["u_H1"][-1], self.hist["p_L2"][-1]))
