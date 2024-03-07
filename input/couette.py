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
    parser.add_argument('-o', '--output', default='couette.cgns', type=str,
        help='name of the output file')
    parser.add_argument('--n_layer', default=16, type=int,
        help='number of layers along the radius')
    parser.add_argument('--n_cell', default=16, type=int,
        help='number of cells along the circle')
    parser.add_argument('--r_inner', default=1.0, type=float,
        help='radius of the inner circle')
    parser.add_argument('--r_outer', default=2.0, type=float,
        help='radius of the outer circle')
    args = parser.parse_args()
    print(args)

    gmsh.initialize(sys.argv)

    # create the line along the X-axis
    gmsh.model.occ.addCircle(x=0, y=0, z=0, r=args.r_inner)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(dim=0),
        size=args.r_inner * 2 * np.pi / args.n_cell)

    # extrude along Z-axis
    input = gmsh.model.getEntities(1)
    print(input)
    output = gmsh.model.occ.extrude(input, dx=0.0, dy=0.0, dz=args.r_inner,
        numElements=[3], heights=[1], recombine=True)
    print(f"{input}.extrude() = {output}")
    gmsh.model.occ.synchronize()
    gmsh.model.geo.synchronize()

    # create the volume by extruding
    n_layer = 16
    numElements = [1] * n_layer
    print("numElements =", numElements)

    h_outer = args.r_outer - args.r_inner
    heights = np.linspace(h_outer / n_layer, h_outer, n_layer)
    input = [(2, 1)]
    output = gmsh.model.geo.extrudeBoundaryLayer(input,
        numElements, -heights, recombine=True)
    print(f"{input}.extrude() = {output}")
    gmsh.model.geo.synchronize()
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(dim=2, tags=[1], name="Inner")
    gmsh.model.addPhysicalGroup(dim=2, tags=[25], name="Outer")
    gmsh.model.addPhysicalGroup(dim=2, tags=[16], name="Front")
    gmsh.model.addPhysicalGroup(dim=2, tags=[24], name="Back")
    gmsh.model.addPhysicalGroup(dim=3, tags=[1], name="Fluid")

    gmsh.model.mesh.generate(3)

    # Launch the GUI to see the model:
    if args.show:
        gmsh.fltk.run()

    gmsh.write(args.output)
    gmsh.finalize()
