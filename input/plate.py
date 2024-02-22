import gmsh
import sys
import numpy as np
import argparse


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
    lx, ly, lz = 0.2, 0.02, 0.1  # length of domain in each direction
    nx, ny, nz = 32, 16, 2 + 1  # number of cells in each direction
    ratio = 1.083317311

    gmsh.initialize(sys.argv)

    # create the line along the X-axis
    gmsh.model.geo.addPoint(0, 0, 0)
    gmsh.model.geo.addPoint(lx, 0, 0)
    gmsh.model.geo.addLine(1, 2)
    gmsh.model.geo.synchronize()

    # create the surface parallel to the X-Y surface
    input = gmsh.model.getEntities(1)
    output = gmsh.model.geo.extrude(input, 0, ly, 0)
    print(f"{input}.extrude() = {output}")
    gmsh.model.geo.synchronize()

    # create the volume by extruding
    numElements = [1] * nz
    heights = np.linspace(lz/nz, lz, nz)
    print("numElements =", numElements)
    print("heights =", heights)
    input = gmsh.model.getEntities(2)
    output = gmsh.model.geo.extrudeBoundaryLayer(input,
        numElements, heights, recombine=True)
    print(f"{input}.extrude() = {output}")
    gmsh.model.geo.synchronize()

    # generate hexahedral mesh
    for tag in (1, 2):
        gmsh.model.mesh.setTransfiniteCurve(tag, nx + 1, "Progression", ratio)
    for tag in (3, 4):
        gmsh.model.mesh.setTransfiniteCurve(tag, ny + 1, "Progression", ratio)
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
