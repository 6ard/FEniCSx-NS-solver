# fenicsx_ns/verification.py

from dataclasses import dataclass
from fenicsx_ns.solvers import NavierStokesSolver
from fenicsx_ns.parameters import TimeParameters
from fenicsx_ns.observers import ErrorNorm
from fenicsx_ns.utils import mesh_size


import numpy as np

@dataclass
class RunResult:
    N : int
    h : float
    dt : float
    errors : dict
    timings : dict | None = None

@dataclass
class ConvergenceResult:
    parameter : str
    values : np.ndarray
    runs : list[RunResult]

    def errors(self, key) :
        return np.array([r.errors[key] for r in self.runs])

    def rates(self, key) :
        return compute_rates(self.values, self.errors(key))

    def table(self, key) :
        pass

def run_case(make_case, N, dt, T, scheme, *, t0=0.0, solver_cls=NavierStokesSolver, **solver_kwargs) : 
    case = make_case(N)
    errors = ErrorNorm(case.exact)
    solver = solver_cls(case=case, time_params=TimeParameters(dt=dt, t0=t0, T=T), scheme=scheme, observers=[errors], **solver_kwargs)
    solver.run()

    # Timings?

    return RunResult(N=N, h=mesh_size(solver.msh), dt=dt, errors=errors.summary(), timings=None)

def convergence_study(parameter, values, run) :
    runs = [run(v) for v in values]
    measured = np.array([getattr(r, parameter) for r in runs])
    return ConvergenceResult(parameter=parameter, values=measured, runs=runs)

def spatial_study(make_case, Ns, dt, T, scheme, **kw) :
    return convergence_study("h", Ns, lambda N: run_case(make_case, N, dt, T, scheme, **kw))

def temporal_study(make_case, N, dts, T, scheme, **kw) :
    return convergence_study("dt", dts, lambda dt: run_case(make_case, N, dt, T, scheme, **kw))

def compute_rates(values, errors) :
    values, errors = np.asarray(values, float), np.asarray(errors, float)
    return np.log(errors[:-1] / errors[1:]) / np.log(values[:-1] / values[1:])

