# fenicsx_ns/cases/meshes.py

import numpy as np
from dolfinx import mesh
from mpi4py import MPI

def create_box_mesh(lower, upper, n, cell_type, comm = MPI.COMM_WORLD):
    if not (len(lower) == len(upper) == len(n)) or len(n) not in (2, 3):
        raise ValueError("Invalid input dimensions for mesh creation.")

    corners = [[float(c) for c in lower], [float(c) for c in upper]]
    cells = [int(c) for c in n]

    if len(n) == 2:
        return mesh.create_rectangle(comm, corners, cells, cell_type)
    return mesh.create_box(comm, corners, cells, cell_type)


def tag_exterior_facets(msh, marker, subdomains=None):
    # Every exterior facet gets marker, except where a locator in subdomains = {marker: locator(x)} is True
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    facets = mesh.exterior_facet_indices(msh.topology)
    values = np.full(facets.size, marker, dtype=np.int32)
    for sub_marker, locator in (subdomains or {}).items():
        values[np.isin(facets, mesh.locate_entities_boundary(msh, fdim, locator))] = sub_marker
    return mesh.meshtags(msh, fdim, facets, values)