"""Warp the mesh by shifting nodes.
"""
import argparse
import vtk
import os
from timeit import default_timer as timer


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


def traverse(folder: str, action: callable):
    """Traverse all the VTU files in a folder and all its subfolders.
    """
    for filename in os.listdir(folder):
        f = os.path.join(folder, filename)
        if os.path.isfile(f):
            if f[-4:] == '.vtu':
                action(f)
        else:
            traverse(f, action)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'python3 warp_vtu.py',
        description = 'Warp the mesh by shifting nodes.')
    parser.add_argument('--folder',
        default='./', type=str,
        help='which folder to be traversed')
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
    def warp_serial(filename):
        print(filename)
        warp(filename, args.component, +args.scale, axis)
        warp(filename, args.component, -args.scale, axis)
    start = timer()
    traverse(args.folder, lambda f: warp_serial(f))
    end = timer()
    print('The serial version costs', end - start, 'sec.')
