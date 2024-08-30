"""Warp the mesh by shifting nodes.
"""
import argparse
import vtk
import os
from timeit import default_timer as timer
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


def warp_by_host(filename: str, component: str, scale: float, axis: int):
    """Warp a single VTU file by CPU.
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


def warp_by_device(filename: str, component: str, scale: float, axis: int):
    """Warp a single VTU file by GPU.
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
    # read data for shifting
    pairs_on_host = np.ndarray((n_point, 2))
    for p in range(n_point):
        pairs_on_host[p][0] = vtk_points_data.GetTuple(p)[axis]
        pairs_on_host[p][1] = data_array.GetTuple(p)[0]
    # shift coords by GPU
    module = SourceModule("""
        __global__ void shift(double *pairs, double scale, int len) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < len) {
                double *pair = pairs + idx * 2;
                pair[0] += scale * pair[1];
            }
        }
        """)
    func = module.get_function("shift")
    pairs_on_device = cuda.mem_alloc(pairs_on_host.nbytes)
    cuda.memcpy_htod(pairs_on_device, pairs_on_host)
    block_size = (1024, 1, 1)
    grid_size = ((n_point + 1023) // 1024, 1, 1)
    # print(block_size, grid_size)
    func(pairs_on_device, np.float64(scale), np.int32(n_point),
        block=block_size, grid=grid_size)
    cuda.memcpy_dtoh(pairs_on_host, pairs_on_device)
    # update coords in grid
    for p in range(n_point):
        vtk_points_data.SetComponent(p, axis, pairs_on_host[p][0])
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
    parser.add_argument('--mode',
        choices=['cpu', 'gpu', 'mpi'],
        default='cpu',
        help='which parallel mode to apply')
    args = parser.parse_args()
    print(args)
    axis = 0
    if args.axis == 'y':
        axis = 1
    elif args.axis == 'z':
        axis = 2
    else:
        assert  args.axis == 'x'
    start = timer()
    if args.mode == 'cpu':
        traverse(args.folder,
            lambda f: warp_by_host(f, args.component, args.scale, axis))
    elif args.mode == 'gpu':
        traverse(args.folder,
            lambda f: warp_by_device(f, args.component, args.scale, axis))
    elif args.mode == 'mpi':
        pass
    else:
        assert False
    end = timer()
    print(f'The {args.mode} version costs {end - start} sec.')
