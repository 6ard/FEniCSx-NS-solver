# oscillating_cylinder.py
#
# Flow past a cylinder with the inflow U_max sin(2 pi t / period): the fluid leaves through the open boundary in the
# first half of the period and comes back in through it in the second half, as CSF does over a cardiac cycle.
#
# Usage: python oscillating_cylinder.py [backflow]
# backflow is the backflow stabilization parameter (default 1.0, 0 turns it off).
# Writes to output_oscillating/: velocity.bp and pressure.bp (time series for ParaView), the velocity and pressure
# at six times of the simulation, and the forces on the cylinder over time.

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

from fenicsx_ns import (NavierStokesSolver, Progress, Snapshots, StabilizationParameters, TimeParameters, TimeScheme,
                        VTXOutput)
from fenicsx_ns.cases.dfg_cylinder import OscillatingCylinderCase, CYLINDER
from fenicsx_ns.observers import DragLift, PressureDifference
from fenicsx_ns.plotting import plot_snapshots

OUTPUT = "output_oscillating"
pv.OFF_SCREEN = True # Render the snapshots to files instead of a window


if __name__ == "__main__":
    backflow = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    os.makedirs(OUTPUT, exist_ok=True)

    case = OscillatingCylinderCase(U_max=1.5, period=8.0)
    time_params = TimeParameters(t0=0.0, dt=1 / 200, T=case.period)

    forces, pressure = DragLift(CYLINDER, case.U_mean, case.D), PressureDifference(*case.pressure_points)
    snapshots = Snapshots(every=20)
    solver = NavierStokesSolver(case=case, time_params=time_params, scheme=TimeScheme.bdf2(),
                                stabilization=StabilizationParameters(backflow=backflow),
                                observers=[Progress(), forces, pressure, snapshots, VTXOutput(OUTPUT, every=20)])
    solver.run()

    # The same simulation at different times: three during outflow through the open boundary, three during inflow
    times = case.period * np.array([1, 2, 3, 5, 6, 7]) / 8
    plot_snapshots(snapshots, times, "velocity", clim=(0, 1.5 * case.U_max)).screenshot(f"{OUTPUT}/velocity.png")
    plot_snapshots(snapshots, times, "pressure").screenshot(f"{OUTPUT}/pressure.png")

    t = np.asarray(forces.history["t"])
    series = [("Inflow velocity $U(t)$", [case.inflow_amplitude(ti) for ti in t]),
              ("Drag coefficient $C_D$", forces.history["C_D"]),
              ("Lift coefficient $C_L$", forces.history["C_L"]),
              ("Pressure difference $\\Delta p$", pressure.history["dp"])]

    fig, axes = plt.subplots(len(series), 1, figsize=(8, 9), sharex=True, layout="constrained")
    fig.suptitle(f"Oscillating flow past a cylinder, backflow stabilization {backflow:g}")
    for ax, (label, values) in zip(axes, series):
        ax.plot(t, values)
        ax.axvline(case.period / 2, color="gray", linestyle="--", linewidth=1) # The flow reverses here
        ax.set_ylabel(label)
        ax.grid()
    axes[-1].set_xlabel("Time $t$")
    fig.savefig(f"{OUTPUT}/forces.png")
