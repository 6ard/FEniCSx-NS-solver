# fenicsx_ns

import numpy as np
from dolfinx import fem, default_scalar_type
import ufl
from mpi4py import MPI

def global_integral(form, comm):
    return comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM)

def constant(msh, value):
    return fem.Constant(msh, default_scalar_type(value))

def domain_volume(msh):
    dx = ufl.Measure("dx", domain=msh)
    return float(np.real(global_integral(fem.form(constant(msh, 1.0) * dx), msh.comm)))

def subtract_mean(f, integral_form, volume):
    mean = global_integral(integral_form, f.function_space.mesh.comm) / volume
    f.x.array[:] -= mean
    return mean