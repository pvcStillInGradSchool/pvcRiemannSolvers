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
    parser.add_argument('-o', '--output', default='cavity.cgns', type=str,
        help='name of the output file')
    parser.add_argument('--n_cell', default=8, type=int,
        help='number of cells along each axis')
    parser.add_argument('--ratio', default=0.5, type=float,
        help='ratio between sizes of neighboring cells')
    args = parser.parse_args()
    print(args)

    gmsh.initialize(sys.argv)

    # create the line along the X-axis
    gmsh.model.geo.addPoint(0, 0, 0)
    gmsh.model.geo.addPoint(1, 0, 0)
    gmsh.model.geo.addLine(1, 2)
    gmsh.model.geo.synchronize()

    # create the surface parallel to the X-Y surface
    input = gmsh.model.getEntities(1)
    output = gmsh.model.geo.extrude(input, 0, 1, 0)
    print(f"{input}.extrude() = {output}")
    gmsh.model.geo.synchronize()

    # create the volume by extruding
    input = gmsh.model.getEntities(2)
    output = gmsh.model.geo.extrude(input, 0, 0, 1)
    print(f"{input}.extrude() = {output}")
    gmsh.model.geo.synchronize()

    # generate hexahedral mesh
    for dim, tag in gmsh.model.getEntities(1):
        gmsh.model.mesh.setTransfiniteCurve(tag, numNodes=args.n_cell + 1,
            meshType="Bump", coef=args.ratio)
    for dim, tag in gmsh.model.getEntities(2):
        gmsh.model.mesh.setTransfiniteSurface(tag)
        gmsh.model.mesh.setRecombine(dim, tag)
    for dim, tag in gmsh.model.getEntities(3):
        gmsh.model.mesh.setTransfiniteVolume(tag)
        gmsh.model.mesh.setRecombine(dim, tag)

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
