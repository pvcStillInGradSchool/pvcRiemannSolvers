"""Tests for various limiters.
"""
import unittest
import numpy as np
from matplotlib_wrapper import savefig
from matplotlib import pyplot as plt
from matplotlib import axes

import concept
import riemann
import spatial
import detector
import limiter


def smooth(x, k_left=10, k_right=5):
    x_shift = 8
    gauss_width = 1.25
    value = (np.exp(-((x - x_shift) / gauss_width)**2 / 2)
        * np.sin(k_right * x))
    value += (np.exp(-((x + x_shift) / gauss_width)**2 / 2)
        * np.sin(k_left * x))
    return value + 0


def jumps(x):
    a = -np.pi * 3
    b = +np.pi * 3
    sign = np.sign(np.sin(x))
    amplitude = b - x
    if x < 0:
        amplitude = x - a
    return sign * amplitude + 0


class TestLimiters(unittest.TestCase):
    """Test various oscillation limiters.
    """

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        self._x_left = -np.pi * 4
        self._x_right = np.pi * 4
        self._n_element = 63
        self._riemann = riemann.LinearAdvection(1.0)
        self._detector = detector.All()

    def build_scheme(self, method, degree: int) -> spatial.FiniteElement:
        assert issubclass(method, spatial.FiniteElement)
        scheme = method(self._riemann, degree,
            self._n_element, self._x_left, self._x_right)
        return scheme

    def set_xticks(self, axins: axes.Axes, scheme: spatial.FiniteElement,
            xmin, xmax):
        xticks = []
        for i in range(scheme.n_element()):
            cell_i = scheme.get_element_by_index(i)
            x = cell_i.x_left() / cell_i.length()
            if xmin < x and x < xmax:
                xticks.append(x)
        axins.set_xticks(xticks)
        axins.grid(visible=True, axis='x')

    def test_limiters_on_jumps(self):
        degree = 4
        scheme = self.build_scheme(spatial.FRonLegendreRoots, degree)
        u_init = jumps
        limiters = [
            limiter.ZhongXingHui2013(),
            limiter.LiWanAi2020(k_trunc=0),
            limiter.LiWanAi2020(k_trunc=1e0),
            limiter.LiWanAi2020(k_trunc=1e1),
            # limiter.XuXiaoRui2023(alpha=0.1),
            # limiter.XuXiaoRui2023(alpha=1e0),
            # limiter.XuXiaoRui2023(alpha=1e1),
        ]
        markers = ['1', '2', '3', '4', '+']
        _, ax = plt.subplots(figsize=[6, 5])
        axins1 = ax.inset_axes([0.08, 0.55, 0.18, 0.40])
        xmin, xmax = -0.7, +0.7
        axins1.set_xlim(xmin, xmax)
        axins1.set_ylim(-12.5, +13.5)
        self.set_xticks(axins1, scheme, xmin, xmax)
        axins2 = ax.inset_axes([0.75, 0.08, 0.23, 0.30])
        xmin = np.pi * scheme.n_element() / scheme.length() - 1
        xmax = xmin + 2
        axins2.set_xlim(xmin, xmax)
        axins2.set_ylim(-8.0, +7.5)
        self.set_xticks(axins2, scheme, xmin, xmax)
        points = np.linspace(self._x_left, self._x_right, self._n_element * 10)
        x_values = points / scheme.delta_x(0)
        y_values = np.ndarray(len(points))
        for i in range(len(limiters)):
            limiter_i = limiters[i]
            assert isinstance(limiter_i, concept.Limiter)
            scheme.initialize(u_init)
            old_averages = np.ndarray(scheme.n_element())
            for i_element in range(scheme.n_element()):
                old_averages[i_element] = \
                    scheme.get_element_by_index(i_element).expansion().average()
            indices = self._detector.get_troubled_cell_indices(scheme)
            limiter_i.reconstruct(indices, scheme)
            new_averages = np.ndarray(scheme.n_element())
            for i_element in range(scheme.n_element()):
                new_averages[i_element] = \
                    scheme.get_element_by_index(i_element).expansion().average()
            self.assertAlmostEqual(0,
                np.linalg.norm(new_averages - old_averages))
            for k in range(len(points)):
                y_values[k] = scheme.get_solution_value(points[k])
            plt.plot(x_values, y_values, marker=markers[i],
                label=f'{limiter_i.name()}')
            axins1.plot(x_values, y_values, marker=markers[i])
            axins2.plot(x_values, y_values, marker=markers[i])
        scheme.initialize(u_init)
        for i in range(len(points)):
            y_values[i] = u_init(points[i])
        plt.plot(x_values, y_values, '-', label='Exact Solution')
        axins1.plot(x_values, y_values, '-')
        axins2.plot(x_values, y_values, '-')
        ax.indicate_inset_zoom(axins1, edgecolor="gray")
        ax.indicate_inset_zoom(axins2, edgecolor="gray")
        plt.xlabel(r'$x\,/\,h$')
        plt.ylabel(r'$u^h$')
        plt.title(r'$p=$'+f'{degree}, '+r'$h=$'+f'{scheme.delta_x(0):.3f}')
        plt.legend()
        plt.tight_layout()
        # plt.show()
        savefig('compare_limiters_on_jumps')

    def test_limiters_on_smooth(self):
        degree = 4
        scheme = self.build_scheme(spatial.FRonLegendreRoots, degree)
        k1, k2 = 10, 5
        def u_init(x):
            return smooth(x, k1, k2)
        limiters = [
            limiter.ZhongXingHui2013(),
            limiter.LiWanAi2020(k_trunc=0),
            limiter.LiWanAi2020(k_trunc=1e0),
            limiter.LiWanAi2020(k_trunc=1e1),
            # limiter.XuXiaoRui2023(alpha=0.1),
            # limiter.XuXiaoRui2023(alpha=1e0),
            # limiter.XuXiaoRui2023(alpha=1e1),
        ]
        markers = ['1', '2', '3', '4', '+']
        _, ax = plt.subplots(figsize=[6, 5])
        axins = ax.inset_axes([0.32, 0.10, 0.40, 0.25])
        xmin = -9 * scheme.n_element() / scheme.length()
        xmax = -7 * scheme.n_element() / scheme.length()
        axins.set_xlim(xmin, xmax)
        axins.set_ylim(-1.1, +1.1)
        self.set_xticks(axins, scheme, xmin, xmax)
        points = np.linspace(self._x_left, self._x_right, self._n_element * 10)
        x_values = points / scheme.delta_x(0)
        y_values = np.ndarray(len(points))
        for i in range(len(limiters)):
            limiter_i = limiters[i]
            assert isinstance(limiter_i, concept.Limiter)
            scheme.initialize(u_init)
            old_averages = np.ndarray(scheme.n_element())
            for i_element in range(scheme.n_element()):
                old_averages[i_element] = \
                    scheme.get_element_by_index(i_element).expansion().average()
            indices = self._detector.get_troubled_cell_indices(scheme)
            limiter_i.reconstruct(indices, scheme)
            new_averages = np.ndarray(scheme.n_element())
            for i_element in range(scheme.n_element()):
                new_averages[i_element] = \
                    scheme.get_element_by_index(i_element).expansion().average()
            self.assertAlmostEqual(0,
                np.linalg.norm(new_averages - old_averages))
            for k in range(len(points)):
                y_values[k] = scheme.get_solution_value(points[k])
            plt.plot(x_values, y_values, marker=markers[i],
                label=f'{limiter_i.name()}')
            axins.plot(x_values, y_values, marker=markers[i])
        scheme.initialize(u_init)
        for i in range(len(points)):
            y_values[i] = u_init(points[i])
        plt.plot(x_values, y_values, '-', label='Exact Solution')
        axins.plot(x_values, y_values, '-')
        ax.indicate_inset_zoom(axins, edgecolor="gray")
        plt.xlabel(r'$x\,/\,h$')
        plt.ylabel(r'$u^h$')
        plt.title(scheme.name() +
            r', $(\kappa h)_\mathrm{L}=$'+f'{k1*scheme.delta_x(0):.2f}'
            r', $(\kappa h)_\mathrm{R}=$'+f'{k2*scheme.delta_x(0):.2f}')
        plt.legend()
        plt.tight_layout()
        # plt.show()
        savefig('compare_limiters_on_smooth')


if __name__ == '__main__':
    unittest.main()
