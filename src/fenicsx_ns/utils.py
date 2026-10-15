# fenicsx_ns

import numpy as np
from dolfinx import fem, default_scalar_type, cpp
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

def mesh_size(msh, reduction="max") :
    tdim = msh.topology.dim
    num_owned = msh.topology.index_map(tdim).size_local
    h_local = cpp.mesh.h(msh._cpp_object, tdim, np.arange(num_owned, dtype=np.int32))
    if reduction == "max" :
        return msh.comm.allreduce(h_local.max(initial=0.0), op=MPI.MAX)
    if reduction == "min" :
        return msh.comm.allreduce(h_local.min(initial=np.inf), op=MPI.MIN)
    raise ValueError(f"Reduction has to be either \"max\" or \"min\"")