from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeParameters:
        t0: float
        T: float
        dt: float

@dataclass(frozen=True)
class StabilizationParameters:
        grad_div: float = 0.0
        backflow: float = 0.0

@dataclass(frozen=True)
class FluidParameters:
        nu: float
        rho: float

        @property
        def mu(self):
            return self.nu * self.rho