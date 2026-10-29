# fenicsx_ns/observers.py
from __future__ import annotations

from abc import ABC

import numpy as np
import ufl
from dolfinx import fem, geometry, io
from mpi4py import MPI
from ufl import grad, inner
from fenicsx_ns.utils import global_integral, constant, domain_volume, subtract_mean

import time
import sys


class Observer(ABC):
    def on_start(self, solver):
        pass

    def on_step(self, solver):
        pass

    def on_end(self, solver):
        pass

class Progress(Observer):
    def __init__(self, suffix="", length=50, fill="█"):
        self.suffix, self.length, self.fill = suffix, length, fill

    def on_start(self, solver):
        self.prefix = solver.case.__class__.__name__ + " Progress:"

        self.total = solver.num_steps

    def on_step(self, solver):
        percent = f"{100 * ((solver.step) / float(self.total)):.1f}"
        filled_length = int(self.length * (solver.step + 1) // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        
        # \r prints from the beginning of the line to overwrite previous progress
        sys.stdout.write(f'\r{self.prefix} |{bar}| {percent}% {self.suffix}')
        sys.stdout.flush()   

    def on_end(self, solver):
        sys.stdout.write('\r' + ' ' * (self.length + 50) + '\r')

            

class ErrorNorm(Observer): 
    _NORMS = ["u_L2", "u_H1", "p_L2"]

    def __init__(self, exact, quad_degree=8, relative = True, zero_tol = 1e-14) :
        self.exact, self.quad_degree = exact, quad_degree
        self.relative, self.zero_tol = relative, zero_tol

        self.history = {}

    def on_start(self, solver):
        self._mean_free = solver.pressure_is_mean_free
        dx = solver.dx(degree=self.quad_degree)

        u_ex = self.exact.u_ufl(solver.x_c, solver.t_c)
        p_ex = self.exact.p_ufl(solver.x_c, solver.t_c)
        self.p_ex_mean = constant(solver.msh, 0.0)

        if self._mean_free:
            self.p_ex_int = fem.form(p_ex * dx)
            self.volume = domain_volume(solver.msh)

        p_ref = p_ex - self.p_ex_mean if self._mean_free else p_ex

        e_u, e_p = solver.u_h - u_ex, solver.p_h - p_ref

        self._error_forms = {
            "u_L2"  : fem.form(inner(e_u, e_u) * dx),
            "u_H1" : fem.form(inner(grad(e_u), grad(e_u)) * dx),
            "p_L2" : fem.form(e_p * e_p * dx)
        }

        self._reference_forms = {}

        if self.relative:
            self._reference_forms = {
                "u_L2"  : fem.form(inner(u_ex, u_ex) * dx),
                "u_H1" : fem.form(inner(grad(u_ex), grad(u_ex)) * dx),
                "p_L2" : fem.form(p_ref * p_ref * dx)
            }

        keys = ["t", *self._NORMS]
        if self.relative:
            keys += [f"{k}_rel" for k in self._NORMS]
        self.history = {key: [] for key in keys}

    def _norm(self, solver, f) :
        return float(np.sqrt(max(np.real(global_integral(f, solver.msh.comm)), 0.0)))

    def on_step(self, solver) :
        if self._mean_free:
            self.p_ex_mean.value = global_integral(self.p_ex_int, solver.msh.comm) / self.volume

        self.history["t"].append(solver.t)
        for key, form in self._error_forms.items():
            error = self._norm(solver, form)
            self.history[key].append(error)
            if self.relative:
                reference = self._norm(solver, self._reference_forms[key])
                self.history[f"{key}_rel"].append(error / reference if reference > self.zero_tol else error)

    def summary(self) :
        out = {}
        for key, values in self.history.items():
            if key == "t" or not values:
                continue
            arr = np.asarray(values)
            out[key] = float(arr[-1])
            out[f"{key}_max"] = float(np.nanmax(arr) if np.isfinite(arr).any() else np.nan)
        return out

    def on_end(self, solver) : 
        s = self.summary()
        msg = "Final errors: " + ", ".join(f"{k} = {s[k]:.3e}" for k in self._NORMS)
        if self.relative:
            msg += " | relative: " + ", ".join(f"{k} = {s[f'{k}_rel']:.3e}" for k in self._NORMS)

        pass
        #print(msg)

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
        self._mean_free = solver.pressure_is_mean_free

        if self.exact is not None: 
            self.u_ex, self.e_u = fem.Function(solver.V, name="Exact Velocity"), fem.Function(solver.V, name="Velocity Error")
            self.p_ex, self.e_p = fem.Function(solver.Q, name="Exact Pressure"), fem.Function(solver.Q, name="Pressure Error")

            v_functions = [solver.u_h, self.u_ex, self.e_u]
            q_functions = [solver.p_h, self.p_ex, self.e_p]

            self._volume = domain_volume(solver.msh)
            self._p_ex_int = fem.form(self.p_ex * solver.dx)

            if self._mean_free:
                subtract_mean(self.p_ex, self._p_ex_int, self._volume)
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

class DragLift(Observer):
    # Drag and lift coefficients C = 2 F / (rho U_mean^2 D) of the force F the fluid exerts on the facets with this marker
    def __init__(self, marker, U_mean, D):
        self.marker, self.U_mean, self.D = marker, U_mean, D

    def on_start(self, solver):
        n = -solver.n # FacetNormal points out of the fluid; the force on the body needs the normal into the fluid
        sigma = 2 * solver.mu_c * ufl.sym(grad(solver.u_h)) - solver.p_h * ufl.Identity(solver.msh.geometry.dim)
        traction = ufl.dot(sigma, n)
        scale = 2 / (solver.rho_c * self.U_mean**2 * self.D)
        ds = solver.ds(self.marker)
        self._forms = {"C_D": fem.form(scale * traction[0] * ds), "C_L": fem.form(scale * traction[1] * ds)}
        self.history = {"t": [], "C_D": [], "C_L": []}

    def on_step(self, solver):
        self.history["t"].append(solver.t)
        for key, form in self._forms.items():
            self.history[key].append(float(np.real(global_integral(form, solver.msh.comm))))

    def summary(self):
        t = np.asarray(self.history["t"])
        out = {}
        for key in ("C_D", "C_L"):
            values = np.asarray(self.history[key])
            i = int(np.argmax(values))
            out.update({key: values[-1], f"{key}_max": values[i], f"t_{key}_max": t[i]})
        return out

class PressureDifference(Observer):
    # p(a) - p(b) for two points a and b in the domain
    def __init__(self, a, b):
        self.points = np.zeros((2, 3))
        self.points[0, :len(a)], self.points[1, :len(b)] = a, b

    def on_start(self, solver):
        msh = solver.msh
        tdim = msh.topology.dim
        tree = geometry.bb_tree(msh, tdim)
        colliding = geometry.compute_colliding_cells(msh, geometry.compute_collisions_points(tree, self.points), self.points)

        # Evaluate each point in one owned cell per process; a point on a process boundary is averaged
        num_owned = msh.topology.index_map(tdim).size_local
        self._cells = [[c for c in colliding.links(i) if c < num_owned][:1] for i in range(2)]
        counts = [msh.comm.allreduce(len(cells), op=MPI.SUM) for cells in self._cells]
        if 0 in counts:
            raise ValueError(f"Point {self.points[counts.index(0)]} is outside the mesh")
        self._counts = counts
        self.history = {"t": [], "dp": []}

    def _value(self, solver, i):
        cells = self._cells[i]
        local = solver.p_h.eval(self.points[i], np.array(cells, dtype=np.int32))[0] if cells else 0.0
        return solver.msh.comm.allreduce(float(np.real(local)), op=MPI.SUM) / self._counts[i]

    def on_step(self, solver):
        self.history["t"].append(solver.t)
        self.history["dp"].append(self._value(solver, 0) - self._value(solver, 1))

5