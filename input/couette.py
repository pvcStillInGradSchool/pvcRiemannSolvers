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
    parser.add_argument('--n_inner', default=32, type=int,
        help='number of cells along the inner circle')
    parser.add_argument('--r_inner', default=1.0, type=float,
        help='radius of the inner circle')
    parser.add_argument('--r_outer', default=2.0, type=float,
        help='radius of the outer circle')
    args = parser.parse_args()
    print(args)

    gmsh.initialize(sys.argv)

    # create cocentric disks
    outer_tag = gmsh.model.occ.addDisk(xc=0, yc=0, zc=0,
        rx=args.r_outer, ry=args.r_outer)
    inner_tag = gmsh.model.occ.addDisk(xc=0, yc=0, zc=0,
        rx=args.r_inner, ry=args.r_inner)
    gmsh.model.occ.synchronize()
    gmsh.model.geo.synchronize()

    # create the surface by boolean cutting
    outDimTags, outDimTagsMap = gmsh.model.occ.cut(objectDimTags=[(2, outer_tag)], toolDimTags=[(2, inner_tag)], removeTool=True)
    print(outDimTags, outDimTagsMap)
    gmsh.model.occ.synchronize()
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.setSize(gmsh.model.getEntities(dim=0),
        size=args.r_inner * 2 * np.pi / args.n_inner)

    # create the volume by extruding along the Z-axis
    input = gmsh.model.getEntities(2)
    print(input)
    output = gmsh.model.occ.extrude(input, dx=0.0, dy=0.0, dz=args.r_inner,
        numElements=[3], heights=[1], recombine=True)
    print(f"{input}.extrude() = {output}")
    gmsh.model.occ.synchronize()
    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(dim=2, tags=[2], name="Outer")
    gmsh.model.addPhysicalGroup(dim=2, tags=[3], name="Inner")
    gmsh.model.addPhysicalGroup(dim=2, tags=[1], name="Back")
    gmsh.model.addPhysicalGroup(dim=2, tags=[4], name="Front")
    gmsh.model.addPhysicalGroup(dim=3, tags=[1], name="Fluid")

    gmsh.model.mesh.setRecombine(2, 1)
    gmsh.model.mesh.recombine()
    gmsh.model.mesh.optimize()
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(2)

    # Launch the GUI to see the model:
    if args.show:
        gmsh.fltk.run()

    gmsh.write(args.output)
    gmsh.finalize()
