import gmsh
import sys
import numpy as np
import argparse
from thickness_ratio import get_thickness_ratio, coarsen


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python plate.py',
        description = 'Generate the hexahedral mesh for simulating plate boundary layer.',
        epilog = 'Text at the bottom of help')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('-o', '--output', default='plate.cgns', type=str,
        help='name of the output file')
    args = parser.parse_args()
    print(args)

    x, y, z = 0, 0, 0
    l_x, l_y, l_z = 0.2, 0.02, 0.1  # length of domain in each direction
    n_x, n_y, n_z = 64, 64, 2 + 1  # number of cells in each direction
    c_x, c_y = 0, 2  # coarse levels in each direction
    a_x = get_thickness_ratio(0, l_x, 1e-4, n_x)
    a_y = get_thickness_ratio(0, l_y, 1e-5, n_y)
    n_x, a_x = coarsen(c_x, n_x, a_x)
    print(n_x, a_x)
    n_y, a_y = coarsen(c_y, n_y, a_y)
    print(n_y, a_y)

    gmsh.initialize(sys.argv)

    # create the line along the X-axis
    gmsh.model.geo.addPoint(0, 0, 0)
    gmsh.model.geo.addPoint(l_x, 0, 0)
    gmsh.model.geo.addLine(1, 2)
    gmsh.model.geo.synchronize()

    # create the surface parallel to the X-Y surface
    input = gmsh.model.getEntities(1)
    output = gmsh.model.geo.extrude(input, 0, l_y, 0)
    print(f"{input}.extrude() = {output}")
    gmsh.model.geo.synchronize()

    # create the volume by extruding
    numElements = [1] * n_z
    heights = np.linspace(l_z/n_z, l_z, n_z)
    print("numElements =", numElements)
    print("heights =", heights)
    input = gmsh.model.getEntities(2)
    output = gmsh.model.geo.extrudeBoundaryLayer(input,
        numElements, heights, recombine=True)
    print(f"{input}.extrude() = {output}")
    gmsh.model.geo.synchronize()

    # generate hexahedral mesh
    for tag in (1, 2):
        gmsh.model.mesh.setTransfiniteCurve(tag, n_x + 1, "Progression", a_x)
    for tag in (3, 4):
        gmsh.model.mesh.setTransfiniteCurve(tag, n_y + 1, "Progression", a_y)
    gmsh.model.mesh.setTransfiniteSurface(5)
    gmsh.model.mesh.setRecombine(2, 5)

    gmsh.model.addPhysicalGroup(dim=2, tags=[26], name="Left")
    gmsh.model.addPhysicalGroup(dim=2, tags=[18], name="Right")
    gmsh.model.addPhysicalGroup(dim=2, tags=[22], name="Top")
    gmsh.model.addPhysicalGroup(dim=2, tags=[14], name="Bottom")
    gmsh.model.addPhysicalGroup(dim=2, tags=[5], name="Back")
    gmsh.model.addPhysicalGroup(dim=2, tags=[27], name="Front")
    gmsh.model.addPhysicalGroup(dim=3, tags=[1], name="Fluid")

    gmsh.model.mesh.generate(3)

    # Launch the GUI to see the model:
    if args.show:
        gmsh.fltk.run()

    gmsh.write(args.output)
    gmsh.finalize()
