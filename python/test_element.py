"""Test elements for spatial scheme.
"""
import unittest
import numpy as np
from matplotlib_wrapper import savefig
from matplotlib import pyplot as plt

import coordinate
import element
import riemann
from polynomial import Vincent


class TestDGonGaussPoints(unittest.TestCase):
    """Test elements for implementing discontinuous Galerkin schemes.
    """

    def __init__(self, method_name: str = ...) -> None:
        super().__init__(method_name)
        np.random.seed(31415926)
        temp = np.random.rand(2, 2) + np.eye(2)
        self._vector_riemann = riemann.Coupled(
            riemann.LinearAdvection(1 + np.random.rand()),
            riemann.InviscidBurgers(1 + np.random.rand()),
            temp + temp.T)
        self._scalar_riemann = riemann.LinearAdvection(1 + np.random.rand())
        self._elements = []
        c = coordinate.Linear(np.random.rand(), 2 + np.random.rand())
        for method in (element.DGonLegendreRoots, element.DGonLobattoRoots):
            for r in (self._vector_riemann, self._scalar_riemann):
                for p in range(1, 8):
                    self._elements.append(method(r, p, c))

    def test_interior_residuals(self):
        np.random.seed(31415926)
        for e in self._elements:
            assert isinstance(e, element.DGonGaussPoints)
            n_component = e.equation().n_component()
            if n_component == 1:
                e.approximate(lambda x: np.random.rand())
            else:
                e.approximate(lambda x: np.random.rand(n_component))
            expect = element.DiscontinuousGalerkin.get_interior_residual(e)
            actual = e.get_interior_residual()
            self.assertAlmostEqual(0, np.linalg.norm(expect - actual))


class TestFRonGaussPoints(unittest.TestCase):
    """Test elements for implementing flux reconstruction schemes.
    """

    def __init__(self, method_name: str = ...) -> None:
        super().__init__(method_name)
        np.random.seed(31415926)
        temp = np.random.rand(2, 2) + np.eye(2)
        self._vector_riemann = riemann.Coupled(
            riemann.LinearAdvection(1 + np.random.rand()),
            riemann.InviscidBurgers(1 + np.random.rand()),
            temp + temp.T)
        self._scalar_riemann = riemann.LinearAdvection(1 + np.random.rand())
        self._elements = []
        c = coordinate.Linear(np.random.rand(), 2 + np.random.rand())
        for method in (element.FRonLegendreRoots, element.FRonLobattoRoots):
            for r in (self._vector_riemann, self._scalar_riemann):
                for p in range(1, 8):
                    e = method(r, p, c)
                    g = Vincent(p + 1, Vincent.huynh_lumping_lobatto)
                    e.add_correction_function(g)
                    self._elements.append(e)

    def test_interior_residuals(self):
        np.random.seed(31415926)
        for e in self._elements:
            assert isinstance(e, element.FRonGaussPoints)
            n_component = e.equation().n_component()
            if n_component == 1:
                e.approximate(lambda x: np.random.rand())
            else:
                e.approximate(lambda x: np.random.rand(n_component))
            expect = element.LagrangeFR.get_interior_residual(e)
            actual = e.get_interior_residual()
            self.assertAlmostEqual(0, np.linalg.norm(expect - actual))

    def test_interface_residuals(self):
        np.random.seed(31415926)
        for e in self._elements:
            assert isinstance(e, element.FRonGaussPoints)
            n_component = e.equation().n_component()
            if n_component == 1:
                e.approximate(lambda x: np.random.rand())
            else:
                e.approximate(lambda x: np.random.rand(n_component))
            upwind_flux_left = np.random.rand(n_component)
            upwind_flux_right = np.random.rand(n_component)
            expect = np.zeros((e.n_term(), n_component))
            element.LagrangeFR.add_interface_residual(e,
                upwind_flux_left, upwind_flux_right, expect)
            actual = np.zeros((e.n_term(), n_component))
            e.add_interface_residual(
                upwind_flux_left, upwind_flux_right, actual)
            self.assertAlmostEqual(0, np.linalg.norm(expect - actual))


class TestFRonLegendreRoots(unittest.TestCase):
    """Test the element for implementing flux reconstruction schemes.
    """

    def __init__(self, method_name: str = ...) -> None:
        super().__init__(method_name)
        np.random.seed(31415926)
        self._riemann = riemann.LinearAdvection(1 + np.random.rand())
        self._equation = self._riemann.equation()
        self._degree = 4
        self._x_left = 0.0
        self._x_right = np.pi * 2
        self._test_points = np.linspace(self._x_left, self._x_right)
        self._coordinate = coordinate.Linear(self._x_left, self._x_right)
        self._g = Vincent(self._degree + 1, Vincent.huynh_lumping_lobatto)
        self._element = element.FRonLegendreRoots(self._riemann, self._degree,
            self._coordinate)
        self._element.add_correction_function(self._g)
        self._element.approximate(np.sin)

    def test_plot(self):
        """Plot the curves of F^{discontinous} and F^{continous}.
        """
        upwind_flux_left = np.random.rand()
        upwind_flux_right = np.random.rand()
        n_point = 201
        points = np.linspace(self._x_left, self._x_right, n_point)
        dg_flux = np.zeros(n_point)
        fr_flux = np.zeros(n_point)
        for i in range(n_point):
            point_i = points[i]
            dg_flux[i] = self._element.get_dg_flux(point_i)
            fr_flux[i] = self._element.get_fr_flux(
                point_i, upwind_flux_left, upwind_flux_right)
        plt.figure()
        plt.plot(points, dg_flux, 'r--', label='DG Flux')
        plt.plot(points, fr_flux, 'b-', label='FR Flux')
        plt.plot(self._x_left, upwind_flux_left, 'k<',
            label='Upwind Flux At Left')
        plt.plot(self._x_right, upwind_flux_right, 'k>',
            label='Upwind Flux At Right')
        plt.legend()
        # plt.show()
        savefig('RadauFRonLegendreRoots')

    def test_get_dg_flux(self):
        """Test the values of the discontinuous flux.
        """
        for x_global in self._test_points:
            flux_actual = self._element.get_dg_flux(x_global)
            # dg_flux = a * u
            u_approx = self._element.get_solution_value(x_global)
            flux_expect = self._equation.get_convective_flux(u_approx)
            self.assertAlmostEqual(flux_expect, flux_actual)

    def test_get_fr_flux(self):
        """Test the values of the reconstructed continuous flux.
        """
        upwind_flux_left = np.random.rand()
        upwind_flux_right = np.random.rand()
        self.assertAlmostEqual(upwind_flux_left,
            self._element.get_fr_flux(self._x_left,
                upwind_flux_left, upwind_flux_right))
        self.assertAlmostEqual(upwind_flux_right,
            self._element.get_fr_flux(self._x_right,
                upwind_flux_left, upwind_flux_right))
        vincent = Vincent(self._degree + 1, Vincent.huynh_lumping_lobatto)
        for x_global in self._test_points:
            flux_actual = self._element.get_fr_flux(x_global,
                upwind_flux_left, upwind_flux_right)
            # fr_flux = dg_flux + correction
            flux_expect = self._element.get_dg_flux(x_global)
            x_local = self._coordinate.global_to_local(x_global)
            left, right = vincent.local_to_value(x_local)
            flux_expect += left * (upwind_flux_left
                - self._element.get_dg_flux(self._x_left))
            flux_expect += right * (upwind_flux_right
                - self._element.get_dg_flux(self._x_right))
            self.assertAlmostEqual(flux_expect, flux_actual)

    def test_get_fr_flux_gradient(self):
        """Test the gradient of the reconstructed continuous flux.
        """
        upwind_flux_left = np.random.rand()
        upwind_flux_right = np.random.rand()
        points = np.linspace(self._x_left, self._x_right)
        for x_global in points:
            gradient_actual = self._element.get_fr_flux_gradient(x_global,
                upwind_flux_left, upwind_flux_right)
            # 2nd-order finite difference approximation
            delta_x = 0.0001
            flux_right = self._element.get_fr_flux(x_global + delta_x,
                    upwind_flux_left, upwind_flux_right)
            flux_left = self._element.get_fr_flux(x_global - delta_x,
                    upwind_flux_left, upwind_flux_right)
            gradient_approx = (flux_right - flux_left) / (delta_x + delta_x)
            self.assertAlmostEqual(gradient_actual, gradient_approx)


if __name__ == '__main__':
    unittest.main()
