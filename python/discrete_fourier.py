"""Analyze modified wavenumbers for various spatial schemes.
"""
import argparse
import numpy as np
from matplotlib import pyplot as plt

import spatial
import riemann
import temporal
from plot_settings import line_styles


class DiscreteFourierAnalysis:
    """Compare various spatial schemes using discrete Fourier analysis.
    """

    def __init__(self, x_left, x_right, n_element, tau,
            n_sample_per_element) -> None:
        self._a = 1.0
        self._riemann = riemann.LinearAdvection(self._a)
        self._tau = tau
        self._x_left = x_left
        self._x_right = x_right
        self._n_element = n_element
        delta_x = (x_right - x_left) / n_element
        n_sample = self._n_element * n_sample_per_element
        self._w = np.exp(1.0j * 2 * np.pi / n_sample)
        half_sample_gap = (delta_x / n_sample_per_element) / 2
        self._sample_points = np.linspace(self._x_left+half_sample_gap,
            self._x_right-half_sample_gap, n_sample)

    def get_kth_fourier_coeff(self, k, u):
        """Get the kth fourier coefficient of a scalar-valued function u(x).
        """
        n_sample = len(self._sample_points)
        kth_fourier_coeff = 0.0
        for j in range(n_sample):
            x_j = self._sample_points[j]
            u_j = u(x_j)
            kth_fourier_coeff += u_j * self._w**(-j*k)
        return kth_fourier_coeff / n_sample

    def get_wavenumbers(self, spatial_scheme: spatial.FiniteElement):
        """Get the reduced and modified wavenumbers of a FiniteElement scheme.
        """
        k_max = spatial_scheme.n_element() // 2
        k_max *= spatial_scheme.get_element(0.0).n_term()
        reduced_wavenumbers = np.ndarray(k_max)
        modified_wavenumbers = np.ndarray(k_max, complex)
        ode_solver = temporal.RungeKutta(3)
        for k in range(1, 1 + k_max):
            kappa = k * np.pi / (spatial_scheme.length() / 2)
            reduced_wavenumbers[k-1] = kappa * spatial_scheme.delta_x(0)
            def u_init(x):
                return np.exp(1.0j * kappa * x)
            kth_fourier_of_u_init = 0.0j
            kth_fourier_of_u_tau = 0.0j
            # solve the real part
            spatial_scheme.initialize(lambda x: u_init(x).real)
            kth_fourier_of_u_init += self.get_kth_fourier_coeff(k,
                lambda x: spatial_scheme.get_solution_value(x))
            ode_solver.update(spatial_scheme, delta_t=self._tau, t_curr=0.0)
            kth_fourier_of_u_tau += self.get_kth_fourier_coeff(k,
                lambda x: spatial_scheme.get_solution_value(x))
            # solve the imag part
            spatial_scheme.initialize(lambda x: u_init(x).imag)
            kth_fourier_of_u_init += 1j * self.get_kth_fourier_coeff(k,
                lambda x: spatial_scheme.get_solution_value(x))
            ode_solver.update(spatial_scheme, delta_t=self._tau, t_curr=0.0)
            kth_fourier_of_u_tau += 1j * self.get_kth_fourier_coeff(k,
                lambda x: spatial_scheme.get_solution_value(x))
            # put together
            modified_wavenumbers[k-1] = (1.0j * spatial_scheme.delta_x(0)
                / (self._a * self._tau)
                * np.log(kth_fourier_of_u_tau / kth_fourier_of_u_init))
        return reduced_wavenumbers, modified_wavenumbers

    def plot(self, schemes, labels, xticks_ticks, xticks_labels):
        kh_max = xticks_ticks[-1]
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot([0, kh_max], [0, kh_max], '-', label='Exact')
        plt.ylabel(r'$\Re(\tilde{\kappa}h)$')
        plt.xlabel(r'$\kappa h$')
        plt.xticks(xticks_ticks, xticks_labels)
        plt.grid()
        plt.subplot(2,1,2)
        plt.ylabel(r'$\Im(\tilde{\kappa}h)$')
        plt.xlabel(r'$\kappa h$')
        plt.xticks(xticks_ticks, xticks_labels)
        plt.grid()
        plt.plot([0, kh_max], [0, 0], '-', label='Exact')
        for i in range(len(labels)):
            reduced, modified = self.get_wavenumbers(schemes[i])
            # reduced /= (i + 2)
            # modified /= (i + 2)
            plt.subplot(2,1,1)
            plt.plot(reduced, modified.real, label=labels[i],
                linestyle=line_styles[i][1])
            plt.subplot(2,1,2)
            plt.plot(reduced, modified.imag, label=labels[i],
                linestyle=line_styles[i][1])
        plt.subplot(2,1,1)
        plt.legend(loc='upper left')
        plt.subplot(2,1,2)
        plt.legend(loc='lower left')
        plt.tight_layout()

    def compare_degrees(self, method: spatial.FiniteElement):
        """Compare spatial schemes using the same method but different degrees.
        """
        degrees = np.arange(0, 4, step=1, dtype=int)
        schemes = []
        labels = []
        xticks_ticks = [0.0]
        xticks_labels = ['0']
        for degree in degrees:
            schemes.append(method(self._riemann,
                degree, self._n_element, self._x_left, self._x_right))
            order = degree + 1
            labels.append(schemes[-1].name())
            xticks_ticks.append(order * np.pi)
            xticks_labels.append(f'${order}\pi$')
        self.plot(schemes, labels, xticks_ticks, xticks_labels)
        # plt.show()
        plt.savefig(f'compare_{schemes[-1].name(False)}.svg')

    def compare_methods(self, degree: int):
        """Compare spatial schemes using the same degree but different methods.
        """
        methods = [
            spatial.LegendreDG,
            spatial.LagrangeDG,
            spatial.LegendreFR,
            spatial.LagrangeFR,
        ]
        order = degree + 1
        xticks_ticks = [0.0]
        xticks_labels = ['0']
        for p in range(1, order + 1):
            xticks_ticks.append(p * np.pi)
            xticks_labels.append(f'${p}\pi$')
        schemes = []
        labels = []
        for method in methods:
            schemes.append(method(self._riemann,
                degree, self._n_element, self._x_left, self._x_right))
            labels.append(schemes[-1].name())
        self.plot(schemes, labels, xticks_ticks, xticks_labels)
        # plt.show()
        plt.savefig(f'compare_{degree}-degree_methods.svg')

    def compare_all_degrees(self):
        self.compare_degrees(spatial.LegendreDG)
        self.compare_degrees(spatial.LagrangeDG)
        self.compare_degrees(spatial.LegendreFR)
        self.compare_degrees(spatial.LagrangeFR)

    def compare_all_methods(self):
        self.compare_methods(2)
        self.compare_methods(4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='python3 discrete_fourier.py')
    parser.add_argument('-n', '--n_element',
        default=20, type=int,
        help='number of elements')
    parser.add_argument('-s', '--n_sample_per_element',
        default=10, type=int,
        help='number of sample points per element')
    parser.add_argument('-l', '--x_left',
        default=0.0, type=float,
        help='coordinate of the left end of the domain')
    parser.add_argument('-r', '--x_right',
        default=200.0, type=float,
        help='coordinate of the right end of the domain')
    parser.add_argument('-t', '--tau',
        default=0.0001, type=float,
        help='length of the small time range for evaluating the system')
    args = parser.parse_args()
    print(args)
    dfa = DiscreteFourierAnalysis(args.x_left, args.x_right, args.n_element,
        args.tau, args.n_sample_per_element)
    dfa.compare_all_methods()
    dfa.compare_all_degrees() # time-consuming for large degree
