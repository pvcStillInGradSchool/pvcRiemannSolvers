"""Test ODE systems from spatial schemes.
"""
import unittest
import numpy as np
from matplotlib_wrapper import savefig
from matplotlib import pyplot as plt

import spatial
import element
import riemann
from polynomial import Vincent


class TestFRonLegendreRoots(unittest.TestCase):
    """Test the element for implement flux reconstruction schemes.
    """

    def __init__(self, method_name: str = ...) -> None:
        super().__init__(method_name)
        self._x_left = 0.0
        self._x_right = np.pi * 2
        self._n_element = 5
        self._degree = 1
        self._a_const = np.pi
        self._spatial = spatial.FRonLegendreRoots(
            riemann.LinearAdvection(self._a_const), self._degree,
            self._n_element, self._x_left, self._x_right)
        g = Vincent(self._degree + 1, Vincent.huynh_lumping_lobatto)
        self._spatial.add_correction_function(g)

    def test_reconstruction(self):
        """Plot the curves of the approximate solution and the two fluxes.
        """
        kappa = 4 * np.pi / self._spatial.length()
        def u(x):
            return np.sin(kappa * x)
        self._spatial.initialize(u)
        n_point = 201
        points = np.linspace(self._x_left, self._x_right, n_point)
        exact_solution = np.ndarray(n_point)
        actual_solution = np.ndarray(n_point)
        exact_flux = np.ndarray(n_point)
        dg_flux = np.ndarray(n_point)
        fr_flux = np.ndarray(n_point)
        for i in range(n_point):
            point_i = points[i]
            actual_solution[i] = self._spatial.get_solution_value(point_i)
            exact_solution[i] = u(point_i)
            exact_flux[i] = self._spatial.equation().get_convective_flux(
                exact_solution[i])
            dg_flux[i] = self._spatial.equation().get_convective_flux(
                actual_solution[i])
            fr_flux[i] = self._spatial.get_flux_value(point_i)
        plt.figure()
        plt.subplot(2, 1, 1)
        points /= self._spatial.delta_x(0)
        plt.plot(points, exact_solution, 'r-', label='Exact')
        plt.plot(points, actual_solution, 'b+', label='Actual')
        plt.xlabel(r'$x\,/\,h$')
        plt.ylabel('Solution')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(points, exact_flux, 'r-', label='Exact')
        plt.plot(points, dg_flux, 'b+', label='Discontinuous')
        plt.plot(points, fr_flux, 'gx', label='Reconstructed')
        plt.xlabel(r'$x\,/\,h$')
        plt.ylabel('Flux')
        plt.legend()
        plt.tight_layout()
        # plt.show()
        savefig('FRonLegendreRoots')

    def test_resolution(self):
        degree = 4
        scheme = spatial.FRonLegendreRoots(
            riemann.LinearAdvection(self._a_const), degree,
            self._n_element, self._x_left, self._x_right)
        g = Vincent(degree + 1, Vincent.huynh_lumping_lobatto)
        scheme.add_correction_function(g)
        points = np.linspace(scheme.x_left(), scheme.x_right(), 201)
        x_data = points / scheme.delta_x(0)
        plt.figure(figsize=(6, degree*2))
        for k in range(1, 1+degree):
            plt.subplot(degree, 1, k)
            kappa = k * np.pi / scheme.delta_x(0)
            u_exact = lambda x: np.sin(kappa * (x - scheme.x_left()))
            y_data = u_exact(points)
            plt.plot(x_data, y_data, '--', label='Exact')
            scheme.initialize(u_exact)
            for i in range(len(points)):
                y_data[i] = scheme.get_solution_value(points[i])
            plt.plot(x_data, y_data, '-', label=scheme.name())
            plt.xlabel(r'$x\,/\,h$')
            plt.ylabel(f'sin({k}'+r'$\pi x\,/\,h)$')
            plt.legend(loc='upper right')
            plt.grid()
        plt.tight_layout()
        # plt.show()
        savefig('compare_resolutions')

    def _get_spatial_matrices(self, scheme: spatial.DiscontinuousGalerkin, i_cell: int):
        i_prev = i_cell - 1
        i_next = (i_cell + 1) % scheme.n_element()
        cell_prev = scheme.get_element_by_index(i_prev)
        cell_curr = scheme.get_element_by_index(i_cell)
        cell_next = scheme.get_element_by_index(i_next)
        n_term = cell_curr.n_term()
        shape = (n_term, n_term)
        first, last = i_cell*n_term, i_next*n_term
        zeros = np.zeros(n_term)
        s_prev = np.ndarray(shape)
        s_curr = np.ndarray(shape)
        s_next = np.ndarray(shape)
        for k in range(n_term):
            k_only = np.zeros(n_term)
            k_only[k] = 1
            cell_prev.set_solution_coeff(k_only)
            cell_curr.set_solution_coeff(zeros)
            cell_next.set_solution_coeff(zeros)
            residual = scheme.get_residual_column()
            s_prev[:,k] = residual[first : last]
            cell_prev.set_solution_coeff(zeros)
            cell_curr.set_solution_coeff(k_only)
            cell_next.set_solution_coeff(zeros)
            residual = scheme.get_residual_column()
            s_curr[:,k] = residual[first : last]
            cell_prev.set_solution_coeff(zeros)
            cell_curr.set_solution_coeff(zeros)
            cell_next.set_solution_coeff(k_only)
            residual = scheme.get_residual_column()
            s_next[:,k] = residual[first : last]
        return s_prev, s_curr, s_next

    def test_dissipation_matrices(self):
        x_left, x_right = -1.0, 1.0
        n_element = 23
        i_curr = n_element // 2
        for degree in range(2, 10):
            # inviscid
            scheme = spatial.FRonLegendreRoots(
                riemann.LinearAdvection(a_const=1.0),
                degree, n_element, x_left, x_right)
            g = Vincent(degree + 1, Vincent.huynh_lumping_lobatto)
            scheme.add_correction_function(g)
            s_prev, s_curr, s_next = self._get_spatial_matrices(scheme, i_curr)
            # viscosity
            nu = np.random.rand()
            scheme = spatial.FRonLegendreRoots(
                riemann.LinearAdvectionDiffusion(a_const=1.0, b_const=nu),
                degree, n_element, x_left, x_right)
            g = Vincent(degree + 1, Vincent.huynh_lumping_lobatto)
            scheme.add_correction_function(g)
            r_prev, r_curr, r_next = self._get_spatial_matrices(scheme, i_curr)
            # compare
            cell_curr = scheme.get_element_by_index(i_curr)
            assert isinstance(cell_curr, element.FRonLegendreRoots)
            mat_b, mat_c, mat_d, mat_e, mat_f = cell_curr.get_dissipation_matrices()
            mat_d += mat_b - mat_c
            self.assertAlmostEqual(0.0,
                np.linalg.norm((r_curr - s_curr) / nu - mat_d))
            self.assertAlmostEqual(0.0,
                np.linalg.norm((r_prev - s_prev) / nu - mat_e))
            self.assertAlmostEqual(0.0,
                np.linalg.norm((r_next - s_next) / nu - mat_f))
            eigvals = np.linalg.eigvals(mat_d)
            self.assertTrue(eigvals.dtype, float)
            self.assertTrue(eigvals.max() < 0)


if __name__ == '__main__':
    unittest.main()
