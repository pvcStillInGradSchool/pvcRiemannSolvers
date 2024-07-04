"""Tests for various jump detectors.
"""
import unittest
import numpy as np
from scipy import special
from matplotlib_wrapper import savefig
from matplotlib import pyplot as plt

import concept
import coordinate
import riemann
import spatial
import detector
import expansion


markers = ['1', '2', '3', '4', '+', 'x']

def smooth(x, k_left=10, k_right=5):
    x_shift = 6
    gauss_width = 1.25
    value = (np.exp(-((x - x_shift) / gauss_width)**2 / 2)
        * np.sin(k_right * x))
    value += (np.exp(-((x + x_shift) / gauss_width)**2 / 2)
        * np.sin(k_left * x))
    return value + 10


def jumps(x):
    a = -np.pi * 3
    b = +np.pi * 3
    sign = np.sign(np.sin(x))
    amplitude = b - x
    if x < 0:
        amplitude = x - a
    return sign * amplitude + 10


class TestDetectors(unittest.TestCase):
    """Test various jump detectors.
    """

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        self._x_left = -np.pi * 4
        self._x_right = np.pi * 4
        self._x_mid = (self._x_left + self._x_right) / 2
        self._n_element = 73
        self._riemann = riemann.LinearAdvection(1.0)
        self._detectors = [
          detector.Krivodonova2004(),
          detector.LiWanAi2011(),
          detector.ZhuJun2021(),
          detector.LiYanHui2022(),
          detector.Persson2006(),
          detector.Kloeckner2011(),
        ]

    def _n_detector(self) -> int:
        return len(self._detectors)

    def _get_detector(self, i: int) -> concept.Detector:
        return self._detectors[i]

    def _build_scheme(self, method, degree: int, n_element: int,
            r: concept.RiemannSolver) -> spatial.FiniteElement:
        scheme = method(r, degree, n_element, self._x_left, self._x_right)
        return scheme

    def test_kloeckner(self):
        input = [
            (10, lambda x: x > 0, 1.0, 0.88, 1.05),
            (10, lambda x: x * (x > 0), 7.2, 1.67, 1.75),
            (10, lambda x: x * x * (x > 0), 13.3, 2.94, 2.99),
            (10, lambda x: special.eval_legendre(10, x), -6.5, 0.0, 0.0),
            (10, lambda x: np.cos(3 + np.sin(1.3 * x)), 4.2, 4.24, 4.84),
            (10, lambda x: np.sin(np.pi * x), 7.8, 4.12, 4.27),
            (20, lambda x: x > 0.91, 0.8, 0.56, 0.75),
        ]
        coord = coordinate.Linear(-1, 1)
        kloeckner = detector.Kloeckner2011()
        for degree, function, s_raw, s_sl, s_bdsl in input:
            lagrange = expansion.LagrangeOnLobattoRoots(degree, coord)
            lagrange.approximate(function)
            legendre = expansion.Legendre(degree, coord, lagrange.value_type())
            legendre.approximate(lambda x: lagrange.global_to_value(x))
            energy_array = np.ndarray(legendre.n_term(), legendre.value_type())
            min_log10_q = 0
            for k in range(legendre.n_term()):
                energy_array[k] = legendre.get_mode_energy(k)
                min_log10_q = min(min_log10_q, np.log10(energy_array[k])/2)
            s = kloeckner.get_least_square_slope(energy_array)
            # print('Raw', s_raw, s)
            if min_log10_q < -10:
                self.assertAlmostEqual(s_raw, s, delta=1.6)
            else:
                self.assertAlmostEqual(s_raw, s, delta=0.05)
            energy_copy = energy_array.copy()
            kloeckner.apply_skyline(energy_array)
            s = kloeckner.get_least_square_slope(energy_array)
            # print('SL', s_sl, s)
            self.assertAlmostEqual(s_sl, s, places=2)
            energy_array = energy_copy.copy()
            kloeckner.add_modal_decay(energy_array)
            kloeckner.apply_skyline(energy_array)
            s = kloeckner.get_least_square_slope(energy_array)
            # print('BDSL', s_bdsl, s)
            self.assertAlmostEqual(s_bdsl, s, delta=0.13)

    def test_smoothness_on_jumps(self):
        degree = 4
        scheme = self._build_scheme(spatial.LegendreDG, degree,
            self._n_element, self._riemann)
        u_init = jumps
        scheme.initialize(u_init)
        centers = scheme.delta_x(0)/2 + np.linspace(scheme.x_left(),
            scheme.x_right() - scheme.delta_x(0), self._n_element)
        plt.figure(figsize=(6,6))
        plt.subplot(3,1,1)
        points = np.linspace(self._x_left, self._x_right, self._n_element * 10)
        x_values = points / scheme.delta_x(0)
        u_approx = np.ndarray(len(points))
        u_exact = np.ndarray(len(points))
        for i in range(len(points)):
            u_approx[i] = scheme.get_solution_value(points[i])
            u_exact[i] = u_init(points[i])
        plt.plot(x_values, u_exact, 'r-', label='Exact')
        plt.plot(x_values, u_approx, 'g--', label=r'$p=4$')
        plt.legend()
        plt.title('Jumps Approximated by '+scheme.name(False))
        plt.xlabel(r'$x\,/\,h$')
        plt.ylabel(r'$u^h$')
        plt.grid()
        plt.subplot(3,1,(2,3))
        plt.semilogy()
        x_values = centers / scheme.delta_x(0)
        for i in range(self._n_detector()):
            detector_i = self._get_detector(i)
            # assert isinstance(detector_i, detector.SmoothnessBased)
            y_values = detector_i.get_smoothness_values(scheme)
            plt.plot(x_values, y_values, markers[i],
                label=detector_i.name())
        x_values = [
            scheme.x_left() / scheme.delta_x(0),
            scheme.x_right() / scheme.delta_x(0)
        ]
        plt.plot(x_values, [1, 1], label=r'$Smoothness=1$')
        plt.xlabel(r'$x\,/\,h$')
        plt.ylabel(r'$Smoothness$')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        # plt.show()
        savefig('compare_smoothness_on_jumps')

    def test_smoothness_on_smooth(self):
        degree = 4
        scheme = self._build_scheme(spatial.LegendreDG, degree,
            self._n_element, self._riemann)
        k1, k2 = 10, 20
        def u_init(x):
            return smooth(x, k1, k2)
        scheme.initialize(u_init)
        centers = scheme.delta_x(0)/2 + np.linspace(scheme.x_left(),
            scheme.x_right() - scheme.delta_x(0), self._n_element)
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(3,1,1)
        points = np.linspace(self._x_left, self._x_right, self._n_element * 10)
        x_values = points / scheme.delta_x(0)
        u_approx = np.ndarray(len(points))
        u_exact = np.ndarray(len(points))
        for i in range(len(points)):
            u_approx[i] = scheme.get_solution_value(points[i])
            u_exact[i] = u_init(points[i])
        plt.plot(x_values, u_exact, 'r-', label='Exact')
        plt.plot(x_values, u_approx, 'g--', label=r'$p=4$')
        plt.legend()
        k1_h = k1 * scheme.delta_x(0)
        k2_h = k2 * scheme.delta_x(0)
        plt.title(r'$kh=$'+f'({k1_h:.2f}, {k2_h:.2f})' +
            ' Approximated by '+scheme.name(False))
        plt.xlabel(r'$x\,/\,h$')
        plt.ylabel(r'$u^h$')
        plt.grid()
        ax = fig.add_subplot(3,1,(2,3))
        plt.semilogy()
        for i in range(self._n_detector()):
            detector_i = self._get_detector(i)
            # assert isinstance(detector_i, detector.SmoothnessBased)
            y_values = detector_i.get_smoothness_values(scheme)
            plt.plot(centers/scheme.delta_x(0), y_values, markers[i],
                label=detector_i.name())
        x_values = [
            scheme.x_left() / scheme.delta_x(0),
            scheme.x_right() / scheme.delta_x(0)]
        plt.plot(x_values, [1, 1], label=r'$Smoothness=1$')
        plt.xlabel(r'$x\,/\,h$')
        plt.ylabel(r'$Smoothness$')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        # plt.show()
        savefig('compare_smoothness_on_smooth')

    def test_detectors_on_smooth(self):
        degree = 4
        scheme = self._build_scheme(spatial.LegendreDG, degree,
            self._n_element, self._riemann)
        k_max = int((degree+1) * scheme.n_element() / 2)
        kappa_h = np.ndarray(k_max)
        active_counts = np.ndarray((self._n_detector(), k_max))
        for k in range(k_max):
            kappa = (k+1) * 2 * np.pi / scheme.length()
            kappa_h[k] = kappa * scheme.delta_x(0)
            def u_init(x):
                return 1+np.sin(kappa * (x - scheme.x_left()))
            scheme.initialize(u_init)
            for i in range(self._n_detector()):
                detector_i = self._get_detector(i)
                troubled_cell_indices = \
                    detector_i.get_troubled_cell_indices(scheme)
                active_counts[i][k] = len(troubled_cell_indices)
        plt.figure()
        for i in range(self._n_detector()):
            plt.plot(kappa_h/np.pi, active_counts[i]/scheme.n_element()*100,
                f'-{markers[i]}', label=self._get_detector(i).name())
        plt.title(r'On $1+\sin(\kappa x)$ Approximated by '+scheme.name())
        plt.xlabel(r'$\kappa h\,/\,\pi$')
        plt.ylabel('Troubled Cell Count (%)')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        # plt.show()
        savefig('compare_detectors_on_smooth')

    def test_detectors_on_euler(self):
        roe = riemann.Euler(1.4)
        euler = roe.equation()
        def sod(x):
            if x < self._x_mid:
                return euler.primitive_to_conservative(1, 0, 1)
            else:
                return euler.primitive_to_conservative(0.125, 0, 0.1)
        for n_element in (33, 55, 77):
            for degree in range(3, 10):
                scheme = self._build_scheme(spatial.LegendreDG, degree,
                    n_element, roe)
                scheme.set_boundary_values(sod(self._x_left), sod(self._x_right))
                scheme.initialize(sod)
                for i in range(self._n_detector()):
                    detector_i = self._get_detector(i)
                    troubled_cell_indices = \
                        detector_i.get_troubled_cell_indices(scheme)
                    i_mid = scheme.get_element_index(self._x_mid)
                    # print(degree, i_mid, f'{troubled_cell_indices} by {detector_i.name(True)}')
                    self.assertTrue(i_mid in troubled_cell_indices)
                    self.assertTrue(len(troubled_cell_indices) <= 3)


if __name__ == '__main__':
    unittest.main()
