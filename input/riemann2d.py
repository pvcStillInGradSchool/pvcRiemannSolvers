import gmsh
import sys
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = f'python {sys.argv[0]}.py',
        description = 'Generate the hexahedral mesh for simulating the 2d Riemann problem.')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('-o', '--output', default='riemann2d.cgns', type=str,
        help='name of the output file')
    parser.add_argument('--edge_length', default=5e-2, type=float,
        help='average length of edges')
    parser.add_argument('--n_layer', default=1, type=int,
        help='number of layers')
    parser.add_argument('--structured', action='store_true')
    args = parser.parse_args()
    print(args)

    gmsh.initialize(sys.argv)

    # create control points
    gmsh.model.geo.addPoint(0, 0, 0, args.edge_length)  # P1
    gmsh.model.geo.addPoint(1, 0, 0, args.edge_length)  # P2
    gmsh.model.geo.addPoint(1, 1, 0, args.edge_length)  # P3
    gmsh.model.geo.addPoint(0, 1, 0, args.edge_length)  # P4
    c = 0.8
    gmsh.model.geo.addPoint(c, 0, 0, args.edge_length)  # P5
    gmsh.model.geo.addPoint(1, c, 0, args.edge_length)  # P6
    gmsh.model.geo.addPoint(c, 1, 0, args.edge_length)  # P7
    gmsh.model.geo.addPoint(0, c, 0, args.edge_length)  # P8
    gmsh.model.geo.addPoint(c, c, 0, args.edge_length)  # P9

    # create lines
    gmsh.model.geo.addLine(1, 5)  # L1
    gmsh.model.geo.addLine(5, 2)  # L2
    gmsh.model.geo.addLine(2, 6)  # L3
    gmsh.model.geo.addLine(6, 3)  # L4
    gmsh.model.geo.addLine(3, 7)  # L5
    gmsh.model.geo.addLine(7, 4)  # L6
    gmsh.model.geo.addLine(4, 8)  # L7
    gmsh.model.geo.addLine(8, 1)  # L8
    gmsh.model.geo.addLine(9, 5)  # L9
    gmsh.model.geo.addLine(9, 6)  # L10
    gmsh.model.geo.addLine(9, 7)  # L11
    gmsh.model.geo.addLine(9, 8)  # L12

    # create faces
    gmsh.model.geo.addCurveLoop((1, -9, 12, 8))  # S1
    gmsh.model.geo.addCurveLoop((2, 3, -10, 9))  # S2
    gmsh.model.geo.addCurveLoop((10, 4, 5, -11))  # S3
    gmsh.model.geo.addCurveLoop((11, 6, 7, -12))  # S4

    for i in range(4):
        gmsh.model.geo.addPlaneSurface((i + 1,))
    gmsh.model.geo.synchronize()

    # create the volume by extruding
    input = gmsh.model.getEntities(2)
    output = gmsh.model.geo.extrude(input,
        dx=0, dy=0, dz=args.n_layer * args.edge_length,
        numElements=[args.n_layer], heights=[1], recombine=True)
    print(f"{input}.extrude() = {output}")
    gmsh.model.geo.synchronize()

    # generate hexahedral mesh
    for dim, tag in gmsh.model.getEntities(2):
        if args.structured:
            gmsh.model.mesh.setTransfiniteSurface(tag)
        gmsh.model.mesh.setRecombine(dim, tag)
    for dim, tag in gmsh.model.getEntities(3):
        if args.structured:
            gmsh.model.mesh.setTransfiniteVolume(tag)
        gmsh.model.mesh.setRecombine(dim, tag)

    gmsh.model.addPhysicalGroup(dim=2, tags=[33,95], name="Left")
    gmsh.model.addPhysicalGroup(dim=2, tags=[47,69], name="Right")
    gmsh.model.addPhysicalGroup(dim=2, tags=[73,91], name="Top")
    gmsh.model.addPhysicalGroup(dim=2, tags=[21,43], name="Bottom")
    gmsh.model.addPhysicalGroup(dim=2, tags=[1,2,3,4], name="Back")
    gmsh.model.addPhysicalGroup(dim=2, tags=[34,56,78,100], name="Front")
    gmsh.model.addPhysicalGroup(dim=3, tags=[1,2,3,4], name="Fluid")

    gmsh.model.mesh.generate(3)

    # Launch the GUI to see the model:
    if args.show:
        gmsh.fltk.run()

    gmsh.write(args.output)
    gmsh.finalize()
