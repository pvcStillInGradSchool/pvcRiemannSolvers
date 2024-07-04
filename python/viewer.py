"""Plot solutions or errors in the x-t plane.
"""
import sys
import numpy as np
import vtk
import imageio
from matplotlib_wrapper import savefig
from matplotlib import pyplot as plt
from matplotlib import colors


class Viewer:

    def __init__(self, expect_path, actual_path, n_element, scalar_name,
            suffix) -> None:
        self._expect_path = expect_path
        self._actual_path = actual_path
        self._n_element = n_element
        self._scalar_name = scalar_name
        self._suffix = suffix
        self._actual = self._viscosity = None
        if self._scalar_name == 'Viscosity':
            self._viscosity = Viewer.load_viscosity(self._actual_path)
        else:
            self._expect = Viewer.load(self._expect_path, self._scalar_name)
            self._actual = Viewer.load(self._actual_path, self._scalar_name)
            n_point = self._expect.shape[1]
            self._expect_points = np.linspace(0, self._n_element, n_point)
            n_point = self._actual.shape[1]
            self._actual_points = np.linspace(0, self._n_element, n_point)
            self._ymin = min(self._expect.min(), self._actual.min())
            self._ymax = max(self._expect.max(), self._actual.max())
            delta_y = (self._ymax - self._ymin) * 0.05
            self._ymax += delta_y
            self._ymin -= delta_y

    @staticmethod
    def load(path, scalar_name):
        reader = vtk.vtkXMLUnstructuredGridReader()
        n_frame = 101
        udata = None
        for i_frame in range(n_frame):
            reader.SetFileName(f"{path}/Frame{i_frame}.vtu")
            reader.Update()
            pdata = reader.GetOutput().GetPointData()
            assert isinstance(pdata, vtk.vtkPointData)
            u = pdata.GetAbstractArray(scalar_name)
            if not u:
                return None
            assert isinstance(u, vtk.vtkFloatArray)
            if udata is None:
                n_point = u.GetNumberOfTuples()
                udata = np.ndarray((n_frame, n_point))
            for i_point in range(n_point):
                udata[i_frame][i_point] = u.GetTuple1(i_point)
        return udata

    @staticmethod
    def load_viscosity(path):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(f"{path}/Frame0.vtu")
        reader.Update()
        pdata = reader.GetOutput().GetPointData()
        assert isinstance(pdata, vtk.vtkPointData)
        n_array = pdata.GetNumberOfArrays()
        arrays = []
        for i_array in range(n_array):
            name_i = pdata.GetArrayName(i_array)
            if str.startswith(name_i, 'Viscosity'):
                arrays.append(Viewer.load(path, name_i))
        return arrays

    def plot_frame(self, i_frame, suffix='svg'):
        if self._actual is None:
            return
        fig, ax = plt.subplots()
        ax.set_xlabel('Element Index')
        ax.set_ylabel(self._scalar_name)
        ydata = self._expect[i_frame]
        ax.plot(self._expect_points, ydata, 'b--', label='Expect')
        ydata = self._actual[i_frame]
        ax.plot(self._actual_points, ydata, 'r-', label='Actual')
        ax.set_ylim(self._ymin, self._ymax)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        file_name = f'{self._actual_path}/Frame{i_frame}.{suffix}'
        savefig(file_name)
        plt.close()
        return file_name

    def plot_animation(self, fps=10):
        if self._actual is None:
            return
        frames = []
        for i_frame in range(101):
            png_name = self.plot_frame(i_frame, 'png')
            print(png_name, 'done')
            frames.append(imageio.v2.imread(png_name))
        gif_name = 'Animation.gif'
        imageio.mimsave(gif_name, frames, fps=fps, loop=0)
        print(gif_name, 'done')

    def plot_error(self):
        if self._actual is None:
            return
        fig, ax = plt.subplots()
        ax.set_xlabel('Point Index')
        ax.set_ylabel('Time Index')
        zdata = np.abs(self._expect - self._actual)
        mappable = ax.imshow(zdata, origin='lower', cmap='coolwarm',
            norm=colors.LogNorm(vmin=1e-6, vmax=1e0, clip=True))
        fig.colorbar(mappable, location='top', label='Pointwise Errors')
        plt.tight_layout()
        savefig(f'{self._actual_path}/Error.{self._suffix}')

    def plot_viscosity(self):
        if self._viscosity is None:
            return
        n_array = len(self._viscosity)
        fig, ax = plt.subplots(n_array, 1, sharex=True, figsize=(7, 5))
        vmin = 0
        vmax = 0
        images = []
        for i in range(n_array):
            ax[i].set_ylabel('Frame Index')
            zdata = self._viscosity[i]
            vmax = max(vmax, np.max(zdata))
            images.append(ax[i].imshow(zdata, aspect='auto', origin='lower', cmap='coolwarm'))
        ax[-1].set_xlabel('Element Index')
        _, n_x = zdata.shape
        positions = np.arange(0, n_x, n_x // 5, dtype='int')
        labels = positions // (n_x // self._n_element)
        # print(n_x, positions, labels)
        ax[-1].set_xticks(positions, labels)
        norm = colors.Normalize(vmin, vmax)
        for im in images:
            im.set_norm(norm)
        fig.colorbar(images[0], ax=ax, location='right')
        # fig.tight_layout()
        savefig(f'{self._actual_path}/Viscosity.{self._suffix}',
            bbox_inches='tight')


if __name__ == '__main__':
    expect_path = sys.argv[1]
    actual_path = sys.argv[2]
    n_element = int(sys.argv[3])
    scalar_name = sys.argv[4]
    suffix = sys.argv[5]
    viewer = Viewer(expect_path, actual_path, n_element, scalar_name, suffix)
    viewer.plot_frame(100, suffix)
    viewer.plot_animation()
    viewer.plot_error()
    viewer.plot_viscosity()
