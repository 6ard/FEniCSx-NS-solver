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


def create_channel_cylinder_mesh(length, height, centre, radius, h_cylinder, h_outer, markers, order=2,
                                 comm=MPI.COMM_WORLD, rank=0):
    # [0, length] x [0, height] minus a disk, meshed with gmsh. markers = {"inlet", "wall", "outlet", "cylinder": int}.
    # The size grows from h_cylinder on the cylinder to h_outer over a distance of 0.3; order=2 gives curved cells on the cylinder.
    # The cylinder points on the x- and y-axes through the centre are mesh vertices, so point evaluation there is exact.
    import gmsh
    from dolfinx.io import gmsh as gmshio

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    if comm.rank == rank:
        geo = gmsh.model.geo
        corners = [geo.addPoint(x, y, 0) for x, y in ((0, 0), (length, 0), (length, height), (0, height))]
        bottom, outlet, top, inlet = [geo.addLine(corners[i], corners[(i + 1) % 4]) for i in range(4)]

        cx, cy = centre
        c = geo.addPoint(cx, cy, 0)
        ring = [geo.addPoint(cx + radius * np.cos(a), cy + radius * np.sin(a), 0) for a in np.pi / 2 * np.arange(4)]
        arcs = [geo.addCircleArc(ring[i], c, ring[(i + 1) % 4]) for i in range(4)]

        surface = geo.addPlaneSurface([geo.addCurveLoop([bottom, outlet, top, inlet]), geo.addCurveLoop(arcs)])
        geo.synchronize()

        gmsh.model.addPhysicalGroup(2, [surface], 1)
        for name, curves in (("inlet", [inlet]), ("wall", [bottom, top]), ("outlet", [outlet]), ("cylinder", arcs)):
            gmsh.model.addPhysicalGroup(1, curves, markers[name], name=name)

        field = gmsh.model.mesh.field
        distance = field.add("Distance")
        field.setNumbers(distance, "CurvesList", arcs)
        threshold = field.add("Threshold")
        field.setNumber(threshold, "InField", distance)
        field.setNumber(threshold, "SizeMin", h_cylinder)
        field.setNumber(threshold, "SizeMax", h_outer)
        field.setNumber(threshold, "DistMin", 0.0)
        field.setNumber(threshold, "DistMax", 0.3)
        field.setAsBackgroundMesh(threshold)
        for option in ("MeshSizeFromPoints", "MeshSizeFromCurvature", "MeshSizeExtendFromBoundary"):
            gmsh.option.setNumber(f"Mesh.{option}", 0)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(order)

    mesh_data = gmshio.model_to_mesh(gmsh.model, comm, rank, gdim=2)
    gmsh.finalize()
    return mesh_data.mesh, mesh_data.facet_tags