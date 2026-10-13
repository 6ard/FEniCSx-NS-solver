# fenicsx_ns/observers.py
from __future__ import annotations

from abc import ABC

import numpy as np
from dolfinx import fem, io
from mpi4py import MPI
from ufl import grad, inner


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

class VTXOutput(Observer) :
    def __init__(self, directory, exact = None, every=10) :
        self.directory, self.every = directory, every
        self.exact = exact

    def _write(self, solver) :
        if self.exact is not None:
            self.u_ex.interpolate(self.exact.u_np(solver.t))
            self.p_ex.interpolate(self.exact.p_np(solver.t))

            self.e_u.x.array[:] = solver.u_h.x.array - self.u_ex.x.array
            self.e_p.x.array[:] = solver.p_h.x.array - self.p_ex.x.array
        self.w_V.write(solver.t)
        self.w_Q.write(solver.t)

    def _make_writer(self, solver, filename, functions) : 
        return io.VTXWriter(solver.msh.comm, self.directory + "/" + filename, functions)

    def on_start(self, solver) :
        if self.exact is not None: 
            self.u_ex, self.e_u = fem.Function(solver.V, name="Exact Velocity"), fem.Function(solver.V, name="Velocity Error")
            self.p_ex, self.e_p = fem.Function(solver.Q, name="Exact Pressure"), fem.Function(solver.Q, name="Pressure Error")

            v_functions = [solver.u_h, self.u_ex, self.e_u]
            q_functions = [solver.p_h, self.p_ex, self.e_p]
        else:
            v_functions = [solver.u_h]
            q_functions = [solver.p_h]

        self.w_V = self._make_writer(solver, "velocity.bp", v_functions)
        self.w_Q = self._make_writer(solver, "pressure.bp", q_functions)

    def on_step(self, solver) :
        if (solver.step % self.every) == 0:
            self._write(solver)

    def on_end(self, solver) :
        if solver.step % self.every != 0:
            self._write(solver)

        self.w_Q.close()
        self.w_V.close()