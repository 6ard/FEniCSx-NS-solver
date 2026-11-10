# cylinder_benchmark.py
#
# Runs the DFG flow past a cylinder benchmarks and plots drag, lift and pressure difference against the
# published FeatFlow reference results (see DFGCylinderCase and reference/).
#
# Usage: python cylinder_benchmark.py [2D-1 2D-2 2D-3]
# A finished run is stored in output_cylinder/ and reused; delete the .npz file to run it again.

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from fenicsx_ns import NavierStokesSolver, Progress, TimeParameters, TimeScheme
from fenicsx_ns.cases.dfg_cylinder import DFGCylinderCase, CYLINDER
from fenicsx_ns.observers import DragLift, PressureDifference

OUTPUT, REFERENCE = "output_cylinder", "reference"

TIME = {"2D-1": TimeParameters(t0=0.0, dt=1 / 50, T=10.0),
        "2D-2": TimeParameters(t0=0.0, dt=1 / 400, T=12.0),
        "2D-3": TimeParameters(t0=0.0, dt=1 / 400, T=8.0)}

LABELS = {"C_D": "Drag coefficient $C_D$", "C_L": "Lift coefficient $C_L$", "dp": "Pressure difference $\\Delta p$",
          "C_D_max": "max $C_D$", "C_D_min": "min $C_D$", "C_L_max": "max $C_L$", "C_L_min": "min $C_L$",
          "St": "Strouhal number", "t_C_D_max": "$t$(max $C_D$)", "t_C_L_max": "$t$(max $C_L$)", "dp_end": "$\\Delta p(t=8)$"}

SOLVER, PUBLISHED = "#2a78d6", "#eb6834" # One colour per source, in every figure
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
                     "axes.grid": True, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 2, "legend.frameon": False, "figure.dpi": 150})


def run(variant):
    path = f"{OUTPUT}/{variant}.npz"
    if os.path.exists(path):
        return dict(np.load(path))

    case = DFGCylinderCase(variant)
    forces, pressure = DragLift(CYLINDER, case.U_mean, case.D), PressureDifference(*case.pressure_points)
    solver = NavierStokesSolver(case=case, time_params=TIME[variant], scheme=TimeScheme.bdf2(),
                                observers=[Progress(), forces, pressure])
    solver.run()

    history = {key: np.asarray(values) for key, values in (forces.history | pressure.history).items()}
    os.makedirs(OUTPUT, exist_ok=True)
    np.savez(path, **history)
    return history

def load_reference(variant):
    data = np.loadtxt(f"{REFERENCE}/dfg_{variant}.csv", delimiter=",")
    return dict(zip(("t", "C_D", "C_L", "dp"), data.T))

def last_cycle(history):
    # One period of the vortex shedding, from a minimum of C_L to the next, with time starting at 0.
    # The period is averaged over the last (up to) five cycles.
    t, lift = history["t"], history["C_L"]
    minima = np.flatnonzero((lift[1:-1] < lift[:-2]) & (lift[1:-1] <= lift[2:])) + 1
    i, j = minima[-2], minima[-1]
    cycle = {key: values[i:j + 1] for key, values in history.items()}
    cycle["t"] = cycle["t"] - t[i]
    last = minima[-6:]
    return cycle, (t[last[-1]] - t[last[0]]) / (len(last) - 1)

def key_values(variant, history):
    # The quantities the benchmark is compared on, with the keys of DFGCylinderCase.REFERENCE
    if variant == "2D-1":
        return {key: history[key][-1] for key in ("C_D", "C_L", "dp")}
    if variant == "2D-2":
        case = DFGCylinderCase(variant)
        cycle, period = last_cycle(history)
        return {"C_D_max": cycle["C_D"].max(), "C_D_min": cycle["C_D"].min(),
                "C_L_max": cycle["C_L"].max(), "C_L_min": cycle["C_L"].min(), "St": case.D / (case.U_mean * period)}
    i, j = np.argmax(history["C_D"]), np.argmax(history["C_L"])
    return {"C_D_max": history["C_D"][i], "t_C_D_max": history["t"][i],
            "C_L_max": history["C_L"][j], "t_C_L_max": history["t"][j], "dp_end": history["dp"][-1]}

def plot_history(variant, history, reference, title, xlabel="Time $t$", zoom=None):
    # C_D, C_L and dp over time, one panel each. zoom = {key: (ymin, ymax)} limits the y-axis of a panel.
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True, layout="constrained")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)

    for ax, key in zip(axes, ("C_D", "C_L", "dp")):
        ax.plot(reference["t"], reference[key], color=PUBLISHED, label="Published (FeatFlow)")
        ax.plot(history["t"], history[key], color=SOLVER, linestyle=(0, (4, 2)), label="fenicsx_ns")
        ax.set_ylabel(LABELS[key])
        if zoom and key in zoom:
            ax.set_ylim(*zoom[key])

    axes[0].legend(loc="lower right", ncols=2, bbox_to_anchor=(1, 1))
    axes[-1].set_xlabel(xlabel)
    axes[-1].set_xlim(history["t"][0], history["t"][-1])

    fig.savefig(f"{OUTPUT}/cylinder_{variant}.png")
    return fig

def plot_deviation(values):
    # Relative deviation from the published value of every compared quantity. values = {variant: key_values(...)}
    rows = [(variant, key, value, DFGCylinderCase.REFERENCE[variant][key])
            for variant, computed in values.items() for key, value in computed.items()]
    deviation = [100 * (value - published) / abs(published) for _, _, value, published in rows]
    y = -np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.2), layout="constrained")
    fig.suptitle("Flow past a cylinder: deviation from the published values", x=0.01, ha="left", fontsize=12)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.hlines(y, 0, deviation, color=SOLVER, linewidth=2)
    ax.plot(deviation, y, "o", color=SOLVER, markersize=8)

    reach = max(np.abs(deviation))
    for yi, d, (_, _, value, published) in zip(y, deviation, rows):
        ax.annotate(f"{d:+.2f} %   ({value:.4f} vs {published:.4f})", (max(d, 0), yi), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=9, color=MUTED)

    ax.set_yticks(y, [f"{variant}   {LABELS[key]}" for variant, key, _, _ in rows])
    ax.set_xlim(-1.15 * reach, 2.6 * reach)
    ax.set_xlabel("Deviation from the published value [%]")
    ax.grid(axis="y", visible=False)

    fig.savefig(f"{OUTPUT}/cylinder_deviation.png")
    return fig


if __name__ == "__main__":
    variants = sys.argv[1:] or ["2D-1", "2D-2", "2D-3"]
    values = {}

    for variant in variants:
        history = run(variant)
        values[variant] = key_values(variant, history)
        published = DFGCylinderCase.REFERENCE[variant]

        if variant == "2D-1":
            # Stationary flow: the published values are constants, and the start-up is cut off by the zoom
            reference = {"t": history["t"][[0, -1]]} | {key: [published[key]] * 2 for key in ("C_D", "C_L", "dp")}
            zoom = {key: (min(published[key], history[key][-1]) - 0.1 * abs(published[key]),
                          max(published[key], history[key][-1]) + 0.1 * abs(published[key])) for key in ("C_D", "C_L", "dp")}
            plot_history(variant, history, reference, "DFG 2D-1 (Re = 20, stationary): approach to the steady state", zoom=zoom)
        elif variant == "2D-2":
            cycle, _ = last_cycle(history)
            reference, _ = last_cycle(load_reference(variant))
            plot_history(variant, cycle, reference, "DFG 2D-2 (Re = 100, periodic): one vortex shedding cycle",
                         xlabel="Time since the minimum of $C_L$")
        else:
            plot_history(variant, history, load_reference(variant), "DFG 2D-3 (Re = 100, inflow $\\propto \\sin(\\pi t / 8)$)")

        print(variant)
        for key, value in values[variant].items():
            print(f"  {key:>10}: {value:12.6f}   published {published[key]:12.6f}")

    plot_deviation(values)
