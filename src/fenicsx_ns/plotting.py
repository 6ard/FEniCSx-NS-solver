# fenics_ns/plotting.py

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from dolfinx import plot, fem


def _num_components(f) :
    return int(np.prod(f.ufl_shape)) if f.ufl_shape else 1


def _as_linear(f) :
    shape = f.ufl_shape
    element = ("Lagrange", 1, shape) if shape else ("Lagrange", 1)
    f1 = fem.Function(fem.functionspace(f.function_space.mesh, element))
    f1.interpolate(f) 
    return f1

def to_pyvista(f) :
    f1 = _as_linear(f)
    grid = pv.UnstructuredGrid(*plot.vtk_mesh(f1.function_space))
    n_comp = _num_components(f1)
    values = f1.x.array.real.reshape(-1, n_comp)

    if n_comp == 1 :
        grid.point_data[f.name] = values[:, 0]
    else :
        vectors = np.zeros((grid.n_points, 3))
        vectors[:, :n_comp] = values
        grid.point_data[f.name] = vectors
        grid.point_data["magnitude"] = np.linalg.norm(vectors, axis=1)

    return grid

def plot_field(f, plotter = None, cmap="viridis", **mesh_kwargs) :
    grid = to_pyvista(f)
    scalars = "magnitude" if _num_components(f) > 1 else f.name
    plotter = plotter or pv.Plotter()

    if f.function_space.mesh.geometry.dim == 3:
        plotter.add_mesh(grid.outline(), color="black")
        plotter.add_mesh(grid.slice_orthogonal(), scalars=scalars, cmap=cmap, **mesh_kwargs)
    else: 
        plotter.add_mesh(grid, scalars=scalars, cmap=cmap, **mesh_kwargs)
        plotter.view_xy()

    return plotter

def plot_vector(u, factor = 0.1, stride = 1, normal = "z", plotter = None):
    grid = to_pyvista(u)
    if u.function_space.mesh.geometry.dim == 3:
        origin = np.array(grid.center)
        origin["xyz".index(normal)] += 1e-6 * grid.length
        grid = grid.slice(normal=normal, origin=origin)
        if grid.n_points == 0:
            raise ValueError(f"Slice with normal {normal!r} through {origin} is empty")

    points = pv.PolyData(grid.points[::stride])
    points.point_data[u.name] = grid.point_data[u.name][::stride]
    points.point_data["magnitude"] = grid.point_data["magnitude"][::stride]
    arrows = points.glyph(orient=u.name, scale="magnitude", factor=factor)

    plotter = plotter or pv.Plotter()
    plotter.add_mesh(grid, scalars="magnitude", cmap="viridis")
    plotter.add_mesh(arrows, color="red")

    if u.function_space.mesh.geometry.dim == 2:
        plotter.view_xy()

    return plotter

def plot_comparison(numeric, exact, *, cmap="viridis"):
    error = fem.Function(numeric.function_space, name="Error")
    error.x.array[:] = numeric.x.array - exact.x.array

    plotter = pv.Plotter(shape=(1, 3))

    for col, (title, f) in enumerate([("Numeric", numeric), ("Exact", exact), ("Error", error)]):
        plotter.subplot(0, col)
        plot_field(f, plotter=plotter, cmap=cmap)
        plotter.add_text(title, font_size=10)

    plotter.link_views()

    return plotter

def plot_error_norms(error_norms, norms, *, cmap="viridis"):
    for norm in norms:
        fig, ax = plt.subplots()
        ax.set_title(f"Error Norm: {norm}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Error")
        ax.plot(error_norms.history["t"], error_norms.history[norm], label=norm)
        ax.legend()

        plt.show()

def plot_convergence(result, keys, expected=None):
    fig, ax = plt.subplots()
    x = result.values
    ax.set_title(f"Convergence Study: {result.parameter}")

    for key in keys: 
        e = result.errors(key) 
        (line,) = ax.loglog(x, e, "o-", label=key)
        if expected is not None and key in expected:
            p = expected[key]
            ax.loglog(x, e[-1] * (x / x[-1]) ** p, "--", color=line.get_color(), label=f"0({result.parameter}^{p})")

    ax.set_xlabel(result.parameter)
    ax.set_ylabel("Error")
    ax.set_title(f"Convergence in {result.parameter}")
    ax.grid()
    ax.legend()

    plt.show()

    return fig
