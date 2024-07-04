import filter
import sys
import numpy as np
from vtk import *
from matplotlib import pyplot as plt


def read(scalar_name: str, file_name: str):
    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(file_name)
    reader.Update()
    grid = reader.GetOutput()
    pdata = vtkUnstructuredGrid.GetPointData(grid)
    u = pdata.GetAbstractArray(scalar_name)
    n_point = vtkFloatArray.GetNumberOfTuples(u)
    xdata = np.ndarray(n_point)
    udata = np.ndarray(n_point)
    for i_point in range(n_point):
        udata[i_point] = vtkFloatArray.GetTuple1(u, i_point)
        xdata[i_point], _, _ = vtkUnstructuredGrid.GetPoint(grid, i_point)
    return xdata, udata


def write(xdata, udata, scalar_name: str, file_name: str):
    n_point = len(udata)
    vtk_points = vtkPoints()
    vtk_array = vtkFloatArray()
    vtk_array.SetName(scalar_name)
    vtk_array.SetNumberOfComponents(1)
    for i_point in range(n_point):
        vtk_points.InsertNextPoint((xdata[i_point], 0, 0))
        vtk_array.InsertNextValue(udata[i_point])
    grid = vtkUnstructuredGrid()
    grid.SetPoints(vtk_points)
    grid.GetPointData().SetActiveScalars(scalar_name)
    grid.GetPointData().SetScalars(vtk_array)
    writer = vtkXMLDataSetWriter()
    writer.SetInputData(grid)
    writer.SetFileName(file_name)
    writer.SetDataModeToBinary()
    writer.Write()


if __name__ == '__main__':
    scalar_name = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    xdata, old_array = read(scalar_name, input_file)
    f = filter.Selective(True)
    new_array = f.filter(old_array, 1.0)
    write(xdata, new_array, scalar_name, output_file)
    fig = plt.figure()
    plt.plot(xdata, old_array, 'r-', label='Old')
    plt.plot(xdata, new_array, 'b-', label='New')
    plt.legend()
    plt.tight_layout()
    plt.savefig('temp.svg')

    fig = plt.figure()
    theta = np.linspace(np.pi / 16, np.pi, 201)
    plt.plot(theta, f.damping(theta))
    plt.loglog()
    plt.savefig('damping.svg')
