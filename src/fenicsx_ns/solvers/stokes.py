# fenicsx_ns/solvers/stokes.py

from __future__ import annotations

from fenicsx_ns.solvers.navier_stokes import NavierStokesSolver


class StokesSolver(NavierStokesSolver):
    # Time-dependent Stokes: Navier-Stokes without the convection term
    def _convection(self, u, v):
        return 0
