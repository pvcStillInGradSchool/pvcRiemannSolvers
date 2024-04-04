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
    parser.add_argument('-o', '--output', default='cylinder.cgns', type=str,
        help='name of the output file')
    args = parser.parse_args()
    print(args)

    gmsh.initialize(sys.argv)

    C = 10

    # outer box
    outer_tag = gmsh.model.occ.addRectangle(x=-C, y=-C, z=0, dx=C*4, dy=C*2)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(getBoundaryPoints(2, outer_tag), size=C/5)

    # inner box
    inner_tag = gmsh.model.occ.addRectangle(x=-C/2, y=-C/2, z=0, dx=C*2, dy=C)
    gmsh.model.occ.synchronize()
    inner_size = C/10
    gmsh.model.mesh.setSize(getBoundaryPoints(2, inner_tag), size=inner_size)

    outDimTags, outDimTagsMap = gmsh.model.occ.cut(objectDimTags=[(2, outer_tag)], toolDimTags=[(2, inner_tag)], removeTool=False)

    # outer circle
    r = C / 5
    disk_tag = gmsh.model.occ.addDisk(xc=0, yc=0, zc=0, rx=r, ry=r)
    gmsh.model.occ.synchronize()
    disk_size = inner_size / 2
    gmsh.model.mesh.setSize(getBoundaryPoints(2, disk_tag), size=disk_size)
    # circle_dim_tag = gmsh.model.getBoundary([(2, disk_tag)])

    outDimTags, outDimTagsMap = gmsh.model.occ.cut(objectDimTags=[(2, inner_tag)], toolDimTags=[(2, disk_tag)], removeTool=True)
    gmsh.model.occ.synchronize()
    gmsh.model.geo.synchronize()

    # extrude along Z-axis
    input = gmsh.model.getEntities(2)
    print(input)
    output = gmsh.model.occ.extrude(input, dx=0.0, dy=0.0, dz=r * 2,
        numElements=[3], heights=[1], recombine=True)
    print(f"{input}.extrude() = {output}")
    gmsh.model.geo.synchronize()
    gmsh.model.occ.synchronize()

    # inner circle (boundary layer)
    n_layer = 32
    ratio = 0.9
    numElements = [1] * n_layer
    print("numElements =", numElements)
    h_outer = r * 0.5
    h_inner = h_outer / (1 - ratio ** n_layer) * (1 - ratio)
    heights = np.ndarray(n_layer)
    heights[0] = h_inner
    delta_h = h_inner
    for i in range(1, n_layer):
        delta_h *= ratio
        heights[i] = heights[i - 1] + delta_h
    assert heights[-1] == h_outer
    print("heights =", heights, h_outer)
    input = [(2, 12)]
    output = gmsh.model.geo.extrudeBoundaryLayer(input,
        numElements, heights, recombine=True)
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

    gmsh.model.mesh.setRecombine(2, 1)
    gmsh.model.mesh.setRecombine(2, 2)
    gmsh.model.mesh.recombine()
    gmsh.model.mesh.generate(3)

    # Launch the GUI to see the model:
    if args.show:
        gmsh.fltk.run()

    gmsh.write(args.output)

    gmsh.finalize()
