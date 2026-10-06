import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, plot
import pyvista as pv


### TG parameters and initialization

# Fluid parameters
nu = 0.01
k = 2.0 * np.pi # To guarantee periodicity on the unit square, we choose k = 2 * pi
rho = 1.0 
U_0 = 1.0

# Grid 
N = 32

# Time
t_0 = 0.0
T = 10
dx = 0.1


### Taylor Green (TG) exact solution functions

F = lambda t : np.exp(-nu*k**2*t)

u_exact = lambda x, t, U_0: U_0 * np.sin(k*x[0]) * np.cos(k*x[1]) * F(t)
v_exact = lambda x, t, U_0: -U_0 * np.cos(k*x[0]) * np.sin(k*x[1]) * F(t)
velocities_exact = lambda x, t, U_0: np.array([u_exact(x, t, U_0), v_exact(x, t, U_0)])

p_exact = lambda x, t, U_0: rho * U_0**2 / 4 * (np.cos(2*k*x[0]) + np.cos(2*k*x[1])) * F(t)**2

make_TG_domain = lambda N : mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.quadrilateral) 


def main():



    # Initialization
    domain = make_TG_domain(N)

    Q = fem.functionspace(domain, ("Lagrange", 1))
    V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))

    p_exact_function = fem.Function(Q)
    p_exact_function.interpolate(lambda x: p_exact(x, t_0, U_0))

    u_exact_function = fem.Function(V)
    u_exact_function.interpolate(lambda x: velocities_exact(x, t_0, U_0))



    # Use the velocity function space so the mesh points match the P2 degrees of freedom.
    topology, cell_types, geometry = plot.vtk_mesh(u_exact_function.function_space)

    grid = pv.UnstructuredGrid(topology, cell_types, geometry)

    velocity = u_exact_function.x.array.real.reshape((geometry.shape[0], domain.geometry.dim))
    speed = np.linalg.norm(velocity, axis=1)

    grid["velocity"] = velocity
    grid["speed"] = speed

    plotter = pv.Plotter()

    sargs = dict(
        vertical=False,
        width=0.5,             # Takes up 50% of the horizontal screen
        position_x=0.25,       # Starts at 25% from the left (perfectly centered)
        position_y=0.075,       # Slightly offset from the absolute bottom edge
    )

    plotter.add_mesh(
        grid,
        scalars="speed",
        show_edges=True,
        cmap="plasma",
        clim=(0.0, U_0),
        scalar_bar_args=sargs
    )

    plotter.add_text(f"Time: {t_0:.2f}", name="time_label", position="upper_edge")
    plotter.add_text(f"Params: nu={nu:.2f}, k={k:.2f}, rho={rho:.2f}, N={N}", name="parameter_label", position="lower_edge")

    plotter.view_xy()
    plotter.show()
    plotter.screenshot("taylor_green.png")
    plotter.close()

    #Image(filename="taylor_green.png")



    def update_exact_velocity(t):
        u_exact_function.interpolate(lambda x: velocities_exact(x, t, U_0))

    gif = pv.Plotter()



    gif.add_mesh(
        grid,
        scalars="speed",
        show_edges=True,
        cmap="plasma",
        clim=(0.0, U_0),
        scalar_bar_args=sargs
    )

    gif.enable_2d_style()
    gif.view_xy()

    gif.open_gif("taylor_green.gif")
    T = 7
    dx = 0.25

    for t in np.arange(t_0, T, dx):
        update_exact_velocity(t)
        velocity = u_exact_function.x.array.real.reshape((geometry.shape[0], domain.geometry.dim))
        speed = np.linalg.norm(velocity, axis=1)
        grid["speed"] = speed
        gif.add_text(f"Time: {t:.2f}", name="time_label", position="upper_edge")
        gif.add_text(f"Params: nu={nu:.2f}, k={k:.2f}, rho={rho:.2f}, N={N}", name="parameter_label", position="lower_edge")

        gif.write_frame()
    gif.close()
    #Image(filename="taylor_green.gif")


if __name__ == "__main__":
    main()