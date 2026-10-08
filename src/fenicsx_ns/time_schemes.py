from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class TimeScheme:
       name : str
       order : int
       a: tuple[float, float, float]
       e: tuple[float, float]
       startup: TimeScheme | None = None

       @classmethod
       def bdf1(cls) :
              return cls("BDF1", 1, (1.0, -1.0, 0.0), (1.0, 0.0))

       @classmethod
       def bdf2(cls, startup=True):
              return cls("BDF2", 2, (1.5, -2.0, 0.5), (2.0, -1.0),
                        startup=cls.bdf1() if startup else None)