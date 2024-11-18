import gmsh
import sys
import numpy as np
import argparse


def getBoundaryPoints(dim: int, tag: int) -> list[tuple[int, int]]:
    ans = []
    input = [(dim, tag)]
    output = gmsh.model.getBoundary(input, recursive=True)
    for d, t in output:
        if d == 0:
            ans.append((d, t))
    print(f'{input}.getBoundaryPoints() = {ans}')
    return ans


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python cylinder.py',
        description = 'Generate the hexahedral mesh for simulating flows passing a cylinder.',
        epilog = 'Text at the bottom of help')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--output', default='cylinder', type=str,
        help='name of the output file')
    parser.add_argument('--n_layer', default=4, type=int,
        help='number of layers around the cylinder')
    parser.add_argument('--h_ratio', default=0.7, type=float,
        help='thickness ratio between two layers')
    parser.add_argument('--r_large', default=2.0, type=float,
        help='large radius of the boundary layers')
    parser.add_argument('--r_small', default=1.0, type=float,
        help='small radius of the boundary layers')
    parser.add_argument('--c_outer', default=2.0, type=float,
        help='cell size in the outer box')
    parser.add_argument('--c_inner', default=1.0, type=float,
        help='cell size in the inner box')
    parser.add_argument('--n_z', default=1, type=int,
        help='number of layers along the z-axis')
    parser.add_argument('--half_z', default=1.0, type=float,
        help='half of the thickness along the z-axis')
    parser.add_argument('--recombine', default=False, action='store_true',
        help='recombine tetrahedra to hexahedra')
    args = parser.parse_args()
    print(args)

    gmsh.initialize()

    C = 10
    Z = args.half_z

    # outer box
    outer_tag = gmsh.model.occ.addRectangle(x=-C, y=-C, z=-Z, dx=C*4, dy=C*2)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(getBoundaryPoints(2, outer_tag), size=args.c_outer)

    # inner box
    inner_tag = gmsh.model.occ.addRectangle(x=-C/2, y=-C/2, z=-Z, dx=C*2, dy=C)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(getBoundaryPoints(2, inner_tag), size=args.c_inner)

    outDimTags, outDimTagsMap = gmsh.model.occ.cut(objectDimTags=[(2, outer_tag)], toolDimTags=[(2, inner_tag)], removeTool=False)

    # the large circle
    r = args.r_large
    disk_tag = gmsh.model.occ.addDisk(xc=0, yc=0, zc=-Z, rx=r, ry=r)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(getBoundaryPoints(2, disk_tag), size=args.c_inner / 2)
    # circle_dim_tag = gmsh.model.getBoundary([(2, disk_tag)])

    outDimTags, outDimTagsMap = gmsh.model.occ.cut(objectDimTags=[(2, inner_tag)], toolDimTags=[(2, disk_tag)], removeTool=True)
    gmsh.model.occ.synchronize()
    gmsh.model.geo.synchronize()

    # extrude along Z-axis
    input = gmsh.model.getEntities(2)
    print(input)
    output = gmsh.model.occ.extrude(input, dx=0.0, dy=0.0, dz=Z * 2,
        numElements=[args.n_z], heights=[1], recombine=args.recombine)
    print(f"{input}.extrude() = {output}")
    gmsh.model.geo.synchronize()
    gmsh.model.occ.synchronize()

    gmsh.write(args.output + '.step')

    # the small circle (boundary layer by extruding inward)
    n_layer = args.n_layer
    ratio = args.h_ratio
    numElements = [1] * n_layer
    print("numElements =", numElements)
    h_outer = args.r_small
    h_inner = h_outer / (1 - ratio ** n_layer) * (1 - ratio)
    heights = np.ndarray(n_layer)
    heights[0] = h_inner
    delta_h = h_inner
    for i in range(1, n_layer):
        delta_h *= ratio
        heights[i] = heights[i - 1] + delta_h
    np.testing.assert_approx_equal(heights[-1], h_outer)
    print("heights =", heights, h_outer)
    input = [(2, 12)]
    output = gmsh.model.geo.extrudeBoundaryLayer(input,
        numElements, heights, recombine=args.recombine)
    print(f"{input}.extrude() = {output}")
    gmsh.model.geo.synchronize()
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(dim=2, tags=[4], name="Left")
    gmsh.model.addPhysicalGroup(dim=2, tags=[5], name="Right")
    gmsh.model.addPhysicalGroup(dim=2, tags=[6], name="Top")
    gmsh.model.addPhysicalGroup(dim=2, tags=[3], name="Bottom")
    gmsh.model.addPhysicalGroup(dim=2, tags=[1,2,52], name="Back")
    gmsh.model.addPhysicalGroup(dim=2, tags=[11,13,44], name="Front")
    gmsh.model.addPhysicalGroup(dim=2, tags=[53], name="Wall")
    gmsh.model.addPhysicalGroup(dim=3, tags=[1,2,3], name="Fluid")

    if args.recombine:
        gmsh.model.mesh.setRecombine(2, 1)
        gmsh.model.mesh.setRecombine(2, 2)
        gmsh.model.mesh.recombine()

    gmsh.model.mesh.generate(3)

    # Launch the GUI to see the model:
    if args.show:
        gmsh.fltk.run()

    gmsh.write(args.output + '.cgns')

    gmsh.finalize()
