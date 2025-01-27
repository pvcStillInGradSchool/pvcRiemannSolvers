"""Analyze modified wavenumbers for various spatial schemes.
"""
import argparse
import numpy as np
from matplotlib_wrapper import savefig
from matplotlib import pyplot as plt

import concept
import spatial
import riemann
from polynomial import Huynh, Vincent
from matplotlib_wrapper import line_styles


class WaveNumberDisplayer:
    """Plot modified-wavenumbers for various spatial schemes.
    """

    def __init__(self, x_left, x_right, n_element) -> None:
        self._a = 1.0
        self._riemann = riemann.LinearAdvectionDiffusion(self._a, 1e-3, complex)
        self._x_left = x_left
        self._x_right = x_right
        self._n_element = n_element
        self._h = (x_right - x_left) / n_element

    def get_reynolds(self):
        b = self._riemann.equation().get_diffusive_coeff()
        return self._a * self._h / b

    def get_exact_modified_wavenumbers(self, sampled_wavenumbers: np.ndarray):
        n_sample = len(sampled_wavenumbers)
        exact = np.ndarray(n_sample, dtype=complex)
        exact.real = sampled_wavenumbers
        exact.imag = -sampled_wavenumbers**2 / self.get_reynolds()
        return exact

    def build_scheme(self, Method, degree: int, g: concept.Polynomial):
        assert issubclass(Method, spatial.FiniteElement)
        scheme = Method(self._riemann,
            degree, self._n_element, self._x_left, self._x_right)
        if isinstance(scheme, spatial.FluxReconstruction):
            scheme.add_correction_function(g)
        return scheme

    def get_spatial_matrix(self, scheme: spatial.FiniteElement, kappa_h: float):
        """Get the spatial matrix of a FiniteElement scheme.
        """
        assert isinstance(scheme, spatial.FiniteElement)
        # kappa_h = k_int * 2 * np.pi / scheme.length() * scheme.delta_x(0)
        n_term = scheme.degree() + 1
        matrices = np.zeros((self._n_element, n_term, n_term), dtype=complex)
        for col in range(n_term):
            u_tilde = np.zeros(n_term)
            u_tilde[col] = 1
            global_column = np.ndarray(scheme.n_dof(), dtype=complex)
            for i_element in range(scheme.n_element()):
                first = i_element * n_term
                last = first + n_term
                global_column[first:last] = u_tilde * np.exp(1j * i_element * kappa_h)
            scheme.set_solution_column(global_column)
            global_column = scheme.get_residual_column()
            for i_element in range(scheme.n_element()):
                matrix = matrices[i_element]
                assert matrix.shape == (n_term, n_term)
                first = i_element * n_term
                last = first + n_term
                matrix[:, col] = global_column[first:last]
                matrix[:, col] /= np.exp(1j * i_element * kappa_h)
        return matrices[-1]

    def get_spatial_matrix_fast(self, scheme: spatial.FiniteElement, kappa_h: float):
        """Get the spatial matrix of a FiniteElement scheme.

        This version is fast since it uses the fact that
            r_curr(u_curr, u_prev, u_next) = s_curr * u_curr
                                           + s_prev * u_prev
                                           + s_next * u_next
        """
        assert isinstance(scheme, spatial.FiniteElement)
        # kappa_h = k_int * 2 * np.pi / scheme.length() * scheme.delta_x(0)
        n_term = scheme.degree() + 1
        i_curr = self._n_element // 2
        i_prev = i_curr - 1
        i_next = i_curr + 1
        s_curr = np.zeros((n_term, n_term), dtype=complex)
        s_prev = np.zeros((n_term, n_term), dtype=complex)
        s_next = np.zeros((n_term, n_term), dtype=complex)
        first = i_curr * n_term
        last = first + n_term
        global_column = np.ndarray(scheme.n_dof(), dtype=complex)
        for col in range(n_term):
            # build s_curr[:, col] from residual_j(|k⟩, |0⟩, |0⟩)
            global_column[:] = 0
            global_column[i_curr * n_term + col] = 1
            scheme.set_solution_column(global_column)
            global_column = scheme.get_residual_column()
            s_curr[:, col] = global_column[first:last]
            # build s_prev[:, col] from residual_j(|0⟩, |k⟩, |0⟩)
            global_column[:] = 0
            global_column[i_prev * n_term + col] = 1
            scheme.set_solution_column(global_column)
            global_column = scheme.get_residual_column()
            s_prev[:, col] = global_column[first:last]
            # build s_next[:, col] from residual_j(|0⟩, |0⟩, |k⟩)
            global_column[:] = 0
            global_column[i_next * n_term + col] = 1
            scheme.set_solution_column(global_column)
            global_column = scheme.get_residual_column()
            s_next[:, col] = global_column[first:last]
        s_curr += s_prev * np.exp(-1j * kappa_h)
        s_curr += s_next * np.exp(+1j * kappa_h)
        return s_curr

    def get_spatial_matrix_eigvals(self, scheme: spatial.FiniteElement, kappa_h: float):
        """Get the eigenvalues of a scheme at a given wavenumber.
        """
        matrix = self.get_spatial_matrix_fast(scheme, kappa_h)
        return np.linalg.eigvals(matrix)

    def get_modified_wavenumbers(self, scheme: spatial.FiniteElement,
            sampled_wavenumbers: np.ndarray):
        """Get the modified wavenumbers of a scheme at a given set of wavenumbers.
        """
        n_sample = len(sampled_wavenumbers)
        n_term = scheme.degree() + 1
        modified_wavenumbers = np.ndarray((n_sample, n_term), dtype=complex)
        for i_sample in range(n_sample):
            kappa_h = sampled_wavenumbers[i_sample]
            modified_wavenumbers[i_sample, :] = \
                1j * scheme.delta_x(0) / self._a * \
                self.get_spatial_matrix_eigvals(scheme, kappa_h)
        return modified_wavenumbers

    def get_physical_mode(self, sampled_wavenumbers: np.ndarray,
            modified_wavenumbers: np.ndarray):
        n_sample, n_term = modified_wavenumbers.shape
        assert n_sample == len(sampled_wavenumbers)
        physical_eigvals = np.ndarray(n_sample, dtype=complex)
        for i_sample in range(n_sample):
            eigvals = modified_wavenumbers[i_sample]
            # sort the eigvals by their norms
            norms = eigvals.real**2 + eigvals.imag**2
            pairs = np.ndarray(n_term, dtype=[('x', complex), ('y', float)])
            for i_term in range(n_term):
                pairs[i_term] = (eigvals[i_term], norms[i_term])
            pairs.sort(order='y')
            i_interval = int(np.floor(sampled_wavenumbers[i_sample] / np.pi))
            while i_interval > n_term:
                i_interval -= n_term * 2
            while i_interval <= -n_term:
                i_interval += n_term * 2
            assert -n_term < i_interval <= n_term
            if i_interval < 0:
                i_mode = -1 - i_interval
            else:
                i_mode = i_interval - (i_interval == n_term)
            assert 0 <= i_mode < n_term
            physical_eigvals[i_sample] = pairs[i_mode][0]
        return physical_eigvals

    def plot_modified_wavenumbers(self, Method, degree: int,
            g: concept.Polynomial, n_sample: int):
        """Plot the tilde-kappa_h - kappa_h curves for a given scheme.
        """
        print("plot_modified_wavenumbers")
        print("  Method =", Method)
        print("  degree =", degree)
        print("  g =", g)
        print("  n_sample =", n_sample)
        xticks_labels = np.linspace(-degree-1, degree+1, 2*degree+3, dtype=int)
        # xticks_labels = np.linspace(0, degree+1, degree+2, dtype=int)
        xticks_ticks = xticks_labels * np.pi
        kh_min, kh_max = xticks_ticks[0], xticks_ticks[-1]
        sampled_wavenumbers = np.linspace(kh_min, kh_max, n_sample)
        exact = self.get_exact_modified_wavenumbers(sampled_wavenumbers)
        scheme = self.build_scheme(Method, degree, g)
        modified_wavenumbers = self.get_modified_wavenumbers(
            scheme, sampled_wavenumbers)
        physical_eigvals = self.get_physical_mode(sampled_wavenumbers,
            modified_wavenumbers)
        plt.figure(figsize=(6,9))
        plt.subplot(3,1,1)
        plt.title(f'{scheme.name()}\nRe={self.get_reynolds()}')
        plt.ylabel(r'$\Re(\tilde{\kappa}h)$')
        plt.xlabel(r'$\kappa h\,/\,\pi$')
        plt.plot(sampled_wavenumbers, modified_wavenumbers.real, 'k.')
        plt.plot(sampled_wavenumbers, physical_eigvals.real, 'ro', label='Physical')
        plt.plot(sampled_wavenumbers, exact.real, '-', label='Exact')
        plt.xticks(xticks_ticks, xticks_labels)
        plt.grid()
        plt.legend()
        plt.subplot(3,1,2)
        plt.ylabel(r'$\Im(\tilde{\kappa}h)$')
        plt.xlabel(r'$\kappa h\,/\,\pi$')
        plt.plot(sampled_wavenumbers, modified_wavenumbers.imag, 'k.')
        plt.plot(sampled_wavenumbers, physical_eigvals.imag, 'ro', label='Physical')
        plt.plot(sampled_wavenumbers, exact.imag, '-', label='Exact')
        plt.xticks(xticks_ticks, xticks_labels)
        plt.grid()
        plt.legend()
        plt.subplot(3,1,3)
        plt.ylabel(r'$|\tilde{\kappa}h|$')
        plt.xlabel(r'$\kappa h\,/\,\pi$')
        norms = np.sqrt(modified_wavenumbers.real**2 + modified_wavenumbers.imag**2)
        plt.plot(sampled_wavenumbers, norms, 'k.')
        norms = np.sqrt(physical_eigvals.real**2 + physical_eigvals.imag**2)
        plt.plot(sampled_wavenumbers, norms, 'ro', label='Physical')
        norms = np.sqrt(exact.real**2 + exact.imag**2)
        plt.plot(sampled_wavenumbers, norms, '-', label='Exact')
        plt.xticks(xticks_ticks, xticks_labels)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        # plt.show()
        scheme = self.build_scheme(Method, degree, g)
        savefig(f'all_modes_of_{scheme.name(False)}_p={degree}')

    def compare_wave_numbers(self, methods, degrees, degree_to_corrections,
            n_sample: int, compressed: bool, name: str):
        print("compare_wave_numbers")
        print("  methods =", methods)
        print("  degrees =", degrees)
        print("  degree_to_corrections =", degree_to_corrections)
        print("  n_sample =", n_sample)
        print("  compressed =", compressed)
        print("  name =", name)
        if compressed:
            divisor = r'$(N\pi)$'
        else:
            divisor = r'$\pi$'
        plt.figure(figsize=(6,9))
        plt.subplot(2,1,1)
        plt.ylabel(r'$\Re(\tilde{\kappa}h)\,/\,$'+divisor)
        plt.xlabel(r'$\kappa h\,/\,$'+divisor)
        plt.subplot(2,1,2)
        plt.ylabel(r'$\Im(\tilde{\kappa}h)\,/\,$'+divisor)
        plt.xlabel(r'$\kappa h\,/\,$'+divisor)
        i = 0
        for degree in degrees:
            kh_max = (degree + 1) * np.pi
            sampled_wavenumbers = np.linspace(0, kh_max, n_sample)
            scale = (degree * compressed + 1) * np.pi
            for degree_to_correction in degree_to_corrections:
                g = degree_to_correction(degree)
                for method in methods:
                    scheme = self.build_scheme(method, degree, g)
                    modified_wavenumbers = self.get_modified_wavenumbers(
                        scheme, sampled_wavenumbers)
                    physical_eigvals = self.get_physical_mode(
                        sampled_wavenumbers, modified_wavenumbers)
                    plt.subplot(2,1,1)
                    plt.plot(sampled_wavenumbers/scale,
                             physical_eigvals.real/scale,
                             label=scheme.name(), linestyle=line_styles[i][1])
                    plt.subplot(2,1,2)
                    plt.plot(sampled_wavenumbers/scale,
                             physical_eigvals.imag/scale,
                             label=scheme.name(), linestyle=line_styles[i][1])
                    i += 1
        degree = np.max(degrees)
        kh_max = (degree + 1) * np.pi
        sampled_wavenumbers = np.linspace(0, kh_max, n_sample)
        scale = (degree * compressed + 1) * np.pi
        exact = self.get_exact_modified_wavenumbers(sampled_wavenumbers)
        plt.subplot(2,1,1)
        plt.title(f'Re={self.get_reynolds()}')
        plt.plot(sampled_wavenumbers/scale, exact.real/scale,
                 '-', label='Exact')
        plt.grid()
        plt.legend(handlelength=4)
        plt.subplot(2,1,2)
        plt.plot(sampled_wavenumbers/scale, exact.imag/scale,
                 '-', label='Exact')
        plt.grid()
        plt.legend(handlelength=4)
        plt.tight_layout()
        # plt.show()
        savefig(f'{name}')

    def compare_diffusive_schemes(self, method, degree, degree_to_correction,
            n_sample: int, compressed: bool, name: str):
        print("compare_diffusive_schemes")
        print("  method =", method)
        print("  degree =", degree)
        print("  degree_to_correction =", degree_to_correction)
        print("  n_sample =", n_sample)
        print("  compressed =", compressed)
        print("  name =", name)
        if compressed:
            divisor = r'$(N\pi)$'
        else:
            divisor = r'$\pi$'
        plt.figure(figsize=(6,9))
        plt.subplot(2,1,1)
        plt.ylabel(r'$\Re(\tilde{\kappa}h)\,/\,$'+divisor)
        plt.xlabel(r'$\kappa h\,/\,$'+divisor)
        plt.subplot(2,1,2)
        plt.ylabel(r'$\Im(\tilde{\kappa}h)\,/\,$'+divisor)
        plt.xlabel(r'$\kappa h\,/\,$'+divisor)
        i = 0
        g = degree_to_correction(degree)
        scheme = self.build_scheme(method, degree, g)
        kh_max = (degree + 1) * np.pi
        sampled_wavenumbers = np.linspace(0, kh_max, n_sample)
        scale = (degree * compressed + 1) * np.pi
        b_backup = self._riemann.equation()._b
        beta_backup = (riemann.Solver._beta_0, riemann.Solver._beta_1)
        print(b_backup, beta_backup)
        for beta_0 in (0.5, 2.0, 4.0):
            riemann.Solver._beta_0 = beta_0
            for beta_1 in (0.0, 1.0/12, 1.0):
                riemann.Solver._beta_1 = beta_1
                modified_wavenumbers = self.get_modified_wavenumbers(
                    scheme, sampled_wavenumbers)
                physical_eigvals = self.get_physical_mode(
                    sampled_wavenumbers, modified_wavenumbers)
                # minus the value given by LinearAdvection
                self._riemann.equation()._b = 0
                modified_wavenumbers = self.get_modified_wavenumbers(
                    scheme, sampled_wavenumbers)
                physical_eigvals -= self.get_physical_mode(
                    sampled_wavenumbers, modified_wavenumbers)
                self._riemann.equation()._b = b_backup
                plt.subplot(2,1,1)
                plt.plot(sampled_wavenumbers/scale,
                          physical_eigvals.real/scale,
                          label=self._riemann.diffusive_name(),
                          linestyle=line_styles[i][1])
                plt.subplot(2,1,2)
                plt.plot(sampled_wavenumbers/scale,
                          physical_eigvals.imag/scale,
                          label=self._riemann.diffusive_name(),
                          linestyle=line_styles[i][1])
                i += 1
        riemann.Solver._beta_0, riemann.Solver._beta_1 = beta_backup
        print(self._riemann.equation()._b,
            (self._riemann._beta_0, self._riemann._beta_1))
        exact = self.get_exact_modified_wavenumbers(sampled_wavenumbers)
        plt.subplot(2,1,1)
        plt.title(f'{scheme.name()}\nRe = {self.get_reynolds()}')
        plt.plot(sampled_wavenumbers/scale, 0 * exact.real/scale,
                 '-', label='Exact')
        plt.grid()
        plt.legend(handlelength=4)
        plt.subplot(2,1,2)
        plt.plot(sampled_wavenumbers/scale, exact.imag/scale,
                 '-', label='Exact')
        plt.grid()
        plt.legend(handlelength=4)
        plt.tight_layout()
        # plt.show()
        savefig(f'{name}')

    def compare_scheme_orders(self, method, degree, degree_to_correction,
            n_sample: int, compressed: bool, name: str):
        print("compare_scheme_orders")
        print("  method =", method)
        print("  degree =", degree)
        print("  degree_to_correction =", degree_to_correction)
        print("  n_sample =", n_sample)
        print("  compressed =", compressed)
        print("  name =", name)
        plt.figure(figsize=(8,6))
        plt.ylabel(r'$\Delta(\tilde{\kappa}h)=O(h^{p+1})$')
        plt.xlabel(r'$\kappa h$')
        i = 0
        g = degree_to_correction(degree)
        scheme = self.build_scheme(method, degree, g)
        n_element_vec = 4 ** np.arange(1, 8)
        print(n_element_vec)
        sampled_wavenumbers = 2 * np.pi / n_element_vec
        print(sampled_wavenumbers)
        b_backup = self._riemann.equation()._b
        beta_backup = (riemann.Solver._beta_0, riemann.Solver._beta_1)
        self._riemann.equation()._b = 1e+2
        plt.title(f'{scheme.name()}, a/b={self._a / self._riemann.equation()._b}')
        exact = self.get_exact_modified_wavenumbers(sampled_wavenumbers)
        for beta_0 in (0.5, 2.0, 4.0):
            riemann.Solver._beta_0 = beta_0
            for beta_1 in (0.0, 1.0/12, 1.0):
                riemann.Solver._beta_1 = beta_1
                modified_wavenumbers = self.get_modified_wavenumbers(
                    scheme, sampled_wavenumbers)
                eigvals = self.get_physical_mode(sampled_wavenumbers,
                    modified_wavenumbers)
                eigvals -= exact
                norms = np.sqrt(eigvals.real**2 + eigvals.imag**2)
                plt.plot(sampled_wavenumbers, norms,
                    label=self._riemann.diffusive_name(),
                    linestyle=line_styles[i][1], marker='o')
                i += 1
        self._riemann.equation()._b = b_backup
        riemann.Solver._beta_0, riemann.Solver._beta_1 = beta_backup
        for p in range(degree, degree + 3):
            plt.plot([1e-1, 1e-0], [1e-9, 1e-9 * 10**(p+1)], label=r'$p=$'+f'{p}')
        plt.grid()
        plt.legend(handlelength=4)
        plt.loglog()
        plt.tight_layout()
        # plt.show()
        savefig(f'{name}_p={degree}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='python3 wave_number.py')
    parser.add_argument('--task',
        choices=['AllModes', 'DGvsFR', 'HuynhFR', 'CompareDDG', 'CompareOrders'],
        default='AllModes',
        help='task to be run')
    parser.add_argument('-m', '--method',
        choices=['DGonLegendreRoots', 'DGonLobattoRoots', 'DGonUniformRoots', 'LegendreDG',
                 'FRonLegendreRoots', 'FRonLobattoRoots', 'FRonUniformRoots',],
        default='LegendreDG',
        help='method for spatial discretization')
    parser.add_argument('-d', '--degree',
        default=3, type=int,
        help='degree of polynomials for approximation')
    parser.add_argument('-n', '--n_element',
        default=10, type=int,
        help='number of elements')
    parser.add_argument('-s', '--n_sample',
        default=50, type=int,
        help='number of sample points')
    parser.add_argument('-l', '--x_left',
        default=0.0, type=float,
        help='coordinate of the left end of the domain')
    parser.add_argument('-r', '--x_right',
        default=10.0, type=float,
        help='coordinate of the right end of the domain')
    parser.add_argument('-c', '--compressed',
        action='store_true',
        help='whether the range of input wavenumbers is compressed')
    args = parser.parse_args()
    print(args)
    if args.method == 'FRonLegendreRoots':
        SpatialClass = spatial.FRonLegendreRoots
    elif args.method == 'FRonLobattoRoots':
        SpatialClass = spatial.FRonLobattoRoots
    elif args.method == 'FRonUniformRoots':
        SpatialClass = spatial.FRonUniformRoots
    elif args.method == 'DGonLegendreRoots':
        SpatialClass = spatial.DGonLegendreRoots
    elif args.method == 'DGonLobattoRoots':
        SpatialClass = spatial.DGonLobattoRoots
    elif args.method == 'DGonUniformRoots':
        SpatialClass = spatial.DGonUniformRoots
    elif args.method == 'LegendreDG':
        SpatialClass = spatial.LegendreDG
    else:
        assert False
    wnd = WaveNumberDisplayer(args.x_left, args.x_right, args.n_element)
    degree_to_corrections = [
        lambda p: Vincent(p + 1, Vincent.discontinuous_galerkin),
        lambda p: Vincent(p + 1, Vincent.huynh_lumping_lobatto),
        lambda p: Huynh(p + 1, 1),
        lambda p: Huynh(p + 1, 2),
        lambda p: Huynh(p + 1, 3),
        lambda p: Huynh(p + 1, 4),
        lambda p: Huynh(p + 1, 5),
    ]
    if args.task == 'AllModes':
        wnd.plot_modified_wavenumbers(SpatialClass, args.degree,
            Vincent(args.degree + 1, Vincent.huynh_lumping_lobatto), args.n_sample)
    elif args.task == 'DGvsFR':
        wnd.compare_wave_numbers([spatial.LegendreDG,
            spatial.DGonLegendreRoots, spatial.DGonLobattoRoots,
            spatial.FRonLegendreRoots, spatial.FRonUniformRoots,],
            [args.degree], [lambda p: Huynh(p + 1, 2)],
            args.n_sample, args.compressed, args.task)
    elif args.task == 'HuynhFR':
        wnd.compare_wave_numbers([spatial.FRonLobattoRoots],
            [5], degree_to_corrections,
            args.n_sample, args.compressed, args.task)
    elif args.task == 'CompareDDG':
        wnd.compare_diffusive_schemes(spatial.FRonLobattoRoots,
            3, degree_to_corrections[1],
            args.n_sample, args.compressed, args.task)
    elif args.task == 'CompareOrders':
        wnd.compare_scheme_orders(spatial.FRonLobattoRoots,
            args.degree, degree_to_corrections[3],
            args.n_sample, args.compressed, args.task)
    else:
        pass
