"""Warp the mesh by shifting nodes.
"""
import argparse
import vtk


def warp(filename: str, component: str, scale: float, axis: int):
    """Warp a single VTU file.
    """
    # read the given mesh
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    grid = reader.GetOutput()
    assert isinstance(grid, vtk.vtkUnstructuredGrid)
    n_point = grid.GetNumberOfPoints()
    vtk_points = grid.GetPoints()
    assert isinstance(vtk_points, vtk.vtkPoints)
    vtk_points_data = vtk_points.GetData()
    assert isinstance(vtk_points_data, vtk.vtkDataArray)
    point_data = grid.GetPointData()
    data_array = point_data.GetArray(component)
    assert isinstance(data_array, vtk.vtkDataArray)
    assert n_point == data_array.GetNumberOfTuples()
    # shift nodes
    for p in range(n_point):
        coord = vtk_points_data.GetTuple(p)[axis] \
            + scale * data_array.GetTuple(p)[0]
        vtk_points_data.SetComponent(p, axis, coord)
    # write the modified mesh
    writer = vtk.vtkXMLDataSetWriter()
    writer.SetInputData(grid)
    writer.SetFileName(filename)
    writer.SetDataModeToAscii()
    writer.Write()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'python3 warp_vtu.py',
        description = 'Warp the mesh by shifting nodes.')
    parser.add_argument('--filename',
        default='./', type=str,
        help='which vtu file to be warped')
    parser.add_argument('--axis',
        choices=['x', 'y', 'z'],
        default='z',
        help='along which axis to shift nodes')
    parser.add_argument('--component',
        default='Density', type=str,
        help='name of the selected scalar component')
    parser.add_argument('--scale',
        default=+1.0, type=float,
        help='factor to be multiplied with the scalar component')
    args = parser.parse_args()
    print(args)
    axis = 0
    if args.axis == 'y':
        axis = 1
    elif args.axis == 'z':
        axis = 2
    else:
        assert  args.axis == 'x'
    warp(args.filename, args.component, args.scale, axis)
