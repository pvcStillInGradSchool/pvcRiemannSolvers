"""Tests for special polynomials.
"""
import unittest
import numpy as np
from scipy import integrate, special
from matplotlib_wrapper import savefig
from matplotlib import pyplot as plt

from polynomial import Radau, Huynh, Vincent, LagrangeBasis


class TestRadau(unittest.TestCase):
    """Test the Radau class.
    """

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        self._degree = 5
        self._radau = Radau(self._degree)
        assert self._radau.degree() == self._degree

    def test_values_at_ends(self):
        """Test values and -1 and +1.
        """
        self.assertEqual((0.0, 1.0), self._radau.local_to_value(-1.0))
        self.assertEqual((1.0, 0.0), self._radau.local_to_value(+1.0))

    def test_orthogonality(self):
        """Test Radau_{k} ⟂ Polynpmial_{k-2}.
        """
        for k in range(self._degree - 1):
            integral, _ = integrate.quad(lambda x: x**k * self._radau.local_to_value(x)[0],
                -1.0, 1.0)
            self.assertAlmostEqual(0.0, integral)
            integral, _ = integrate.quad(lambda x: x**k * self._radau.local_to_value(x)[1],
                -1.0, 1.0)
            self.assertAlmostEqual(0.0, integral)

    def test_plot(self):
        """Plot the curves of the two Radau polynomials and their derivatives."""
        n_point = 201
        points = np.linspace(-1.0, 1.0, n_point)
        left_values = np.zeros(n_point)
        left_derivatives = np.zeros(n_point)
        right_values = np.zeros(n_point)
        right_derivatives = np.zeros(n_point)
        for i in range(n_point):
            point_i = points[i]
            left_values[i], right_values[i] = self._radau.local_to_value(point_i)
            left_derivatives[i], right_derivatives[i] = self._radau.local_to_gradient(point_i)
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(points, left_values, 'r--', label=r'$g_{+1}(\xi)$')
        plt.plot(points, right_values, 'b-', label=r'$g_{-1}(\xi)$')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(points, left_derivatives, 'r--', label= r'$dg_{+1}/d\xi$')
        plt.plot(points, right_derivatives, 'b-', label= r'$dg_{-1}/d\xi$')
        plt.legend()
        # plt.show()
        savefig('Radau')


class TestHuynh(unittest.TestCase):
    """Test the Huynh class.
    """

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)

    def test_plot(self):
        """Plot the curves of the two Radau polynomials and their derivatives."""
        n_point = 201
        points = np.linspace(-1.0, 1.0, n_point)
        right_values = np.zeros(n_point)
        right_derivatives = np.zeros(n_point)
        degree = 4
        plt.figure()
        for n_lump in range(1, degree + 1):
            huynh = Huynh(degree, n_lump)
            self.assertEqual(huynh.degree(), degree)
            for i in range(n_point):
                point_i = points[i]
                _, right_values[i] = huynh.local_to_value(point_i)
                _, right_derivatives[i] = huynh.local_to_gradient(point_i)
            plt.subplot(1, 2, 1)
            plt.plot(points, right_values, label=huynh.name())
            plt.subplot(1, 2, 2)
            plt.plot(points, right_derivatives, label=huynh.name())
        plt.subplot(1, 2, 1)
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$g$')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\mathrm{d}g/\mathrm{d}\xi$')
        plt.legend()
        plt.tight_layout()
        # plt.show()
        savefig('HuynhByLumping')

    def test_orthogonality(self):
        """Test g(degree, n_lump) ⟂ Polynpmial_{degree - n_lump - 1}.
        """
        for degree in range(1, 8):
            for n_lump in range(1, degree + 1):
                huynh = Huynh(degree, n_lump)
                for p in range(degree - n_lump):
                    integral, _ = integrate.quad(lambda x:
                        special.eval_legendre(p, x) * huynh.local_to_value(x)[1],
                        -1.0, 1.0)
                    self.assertAlmostEqual(0.0, integral)

    def test_symmetry(self):
        """Test g_left(x) = g_right(-x).
        """
        points = -1 + 2 * np.random.rand(1024)
        for degree in range(1, 8):
            for n_lump in range(1, degree + 1):
                huynh = Huynh(degree, n_lump)
                for x in points:
                    left, _ = huynh.local_to_value(x)
                    _, right = huynh.local_to_value(-x)
                    self.assertEqual(left, right)
                    left, _ = huynh.local_to_gradient(x)
                    _, right = huynh.local_to_gradient(-x)
                    self.assertEqual(left, -right)


class TestVincent(unittest.TestCase):
    """Test the Vincent class.
    """

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        self._degree = 5
        self._vincent = Vincent(self._degree, Vincent.huynh_lumping_lobatto)

    def test_radau_equivalence(self):
        """Test Vincent_{k} ≡ Radau_{k+1}, except for the meaning of left/right.
        """
        radau = Radau(self._degree)
        vincent = Vincent(self._degree, Vincent.discontinuous_galerkin)
        self.assertEqual(radau.degree(), vincent.degree())
        points = np.linspace(-1, 1, num=3)
        for x in points:
            radau_left, radau_right = radau.local_to_value(x)
            vincent_left, vincent_right = vincent.local_to_value(x)
            self.assertAlmostEqual(radau_left, vincent_right)
            self.assertAlmostEqual(radau_right, vincent_left)
            radau_left, radau_right = radau.local_to_gradient(x)
            vincent_left, vincent_right = vincent.local_to_gradient(x)
            self.assertAlmostEqual(radau_left, vincent_right)
            self.assertAlmostEqual(radau_right, vincent_left)

    def test_huynh_equivalence(self):
        """Test g_huynh(p, 1) = g_vincent(p, dg) and
                g_huynh(p, 2) = g_vincent(p, fr).
        """
        points = -1 + 2 * np.random.rand(1024)
        for degree in range(2, 8):
            pairs = [
                (Huynh(degree, 1),
                 Vincent(degree, Vincent.discontinuous_galerkin)),
                (Huynh(degree, 2),
                 Vincent(degree, Vincent.huynh_lumping_lobatto)),
            ]
            for huynh, vincent in pairs:
                for x in points:
                    huynh_left, huynh_right = huynh.local_to_value(x)
                    vincent_left, vincent_right = vincent.local_to_value(x)
                    self.assertAlmostEqual(huynh_left, vincent_left)
                    self.assertAlmostEqual(huynh_right, vincent_right)
                    huynh_left, huynh_right = huynh.local_to_gradient(x)
                    vincent_left, vincent_right = vincent.local_to_gradient(x)
                    self.assertAlmostEqual(huynh_left, vincent_left)
                    self.assertAlmostEqual(huynh_right, vincent_right)

    def test_values_at_ends(self):
        """Test values and derivatives and -1 and +1.
        """
        left, right = self._vincent.local_to_value(+1.0)
        self.assertAlmostEqual(0.0, left)
        self.assertAlmostEqual(1.0, right)
        left, right = self._vincent.local_to_value(-1.0)
        self.assertAlmostEqual(0.0, right)
        self.assertAlmostEqual(1.0, left)
        left, right = self._vincent.local_to_value(+1.0)
        self.assertAlmostEqual(0.0, left)
        left, right = self._vincent.local_to_value(-1.0)
        self.assertAlmostEqual(0.0, right)

    def test_orthogonality(self):
        """Test Huynh_{k} ⟂ Polynpmial_{k-3}.
        """
        huynh = Vincent(self._degree, Vincent.huynh_lumping_lobatto)
        for k in range(self._degree - 2):
            integral, _ = integrate.quad(lambda x: x**k * huynh.local_to_value(x)[0],
                -1.0, 1.0)
            self.assertAlmostEqual(0.0, integral)
            integral, _ = integrate.quad(lambda x: x**k * huynh.local_to_value(x)[1],
                -1.0, 1.0)
            self.assertAlmostEqual(0.0, integral)

    def test_plot(self):
        """Plot the curves of the two Radau polynomials and their derivatives."""
        n_point = 201
        points = np.linspace(-1.0, 1.0, n_point)
        left_values = np.zeros(n_point)
        left_derivatives = np.zeros(n_point)
        right_values = np.zeros(n_point)
        right_derivatives = np.zeros(n_point)
        for i in range(n_point):
            point_i = points[i]
            left_values[i], right_values[i] = self._vincent.local_to_value(point_i)
            left_derivatives[i], right_derivatives[i] = self._vincent.local_to_gradient(point_i)
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(points, left_values, 'r--', label=r'$g_{-1}(\xi)$')
        plt.plot(points, right_values, 'b-', label=r'$g_{+1}(\xi)$')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(points, left_derivatives, 'r--', label= r'$dg_{-1}/d\xi$')
        plt.plot(points, right_derivatives, 'b-', label= r'$dg_{+1}/d\xi$')
        plt.legend()
        # plt.show()
        savefig('HuynhByVincent')


class TestLagrangeBasis(unittest.TestCase):
    """Test the LagrangeBasis class.
    """

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        self._n_point = 5
        points = np.linspace(-1.0, 1.0, self._n_point)
        self._lagrange = LagrangeBasis(points)
        assert self._lagrange.degree() == self._n_point - 1

    def test_plot(self):
        """Plot the curves of each Lagrange polynomial.
        """
        n_point = 101
        points = np.linspace(-1.0, 1.0, n_point)
        values = np.zeros((n_point, self._n_point))
        for i in range(n_point):
            point_i = points[i]
            values[i] = self._lagrange.local_to_value(point_i)
        plt.figure()
        for i in range(self._n_point):
            plt.plot(points, values[:, i], label=f'$L_{i}$')
        plt.legend()
        # plt.show()
        savefig('Lagrange')

    def test_delta_property(self):
        """Test the delta_property, i.e. L_i(x_j) = delta_{ij}.
        """
        points = np.linspace(-1.0, 1.0, self._n_point)
        for i in range(self._n_point):
            values = self._lagrange.local_to_value(points[i])
            for j in range(self._n_point):
                if i == j:
                    self.assertEqual(1.0, values[j])
                else:
                    self.assertEqual(0.0, values[j])


if __name__ == '__main__':
    unittest.main()
