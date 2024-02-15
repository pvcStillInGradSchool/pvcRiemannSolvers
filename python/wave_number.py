"""Analyze modified wavenumbers for various spatial schemes.
"""
import argparse
import numpy as np
from matplotlib import pyplot as plt

import concept
import spatial
import riemann
from polynomial import Huynh, Vincent


class WaveNumberDisplayer:
    """Plot modified-wavenumbers for various spatial schemes.
    """

    def __init__(self, x_left, x_right, n_element) -> None:
        self._a = 1.0
        self._riemann = riemann.LinearAdvection(self._a, complex)
        self._x_left = x_left
        self._x_right = x_right
        self._n_element = n_element

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
            global_column = np.ndarray(n_term * scheme.n_element(),
                dtype=complex)
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
        global_column = np.ndarray(n_term * scheme.n_element(),
            dtype=complex)
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

    def get_modified_wavenumbers(self, scheme: spatial.FiniteElement,
            sampled_wavenumbers: np.ndarray):
        """Get the eigenvalues of a scheme at a given set of wavenumbers.
        """
        n_sample = len(sampled_wavenumbers)
        n_term = scheme.degree() + 1
        modified_wavenumbers = np.ndarray((n_sample, n_term), dtype=complex)
        for i_sample in range(n_sample):
            kappa_h = sampled_wavenumbers[i_sample]
            matrix = self.get_spatial_matrix_fast(scheme, kappa_h)
            matrix *= 1j * scheme.delta_x(0) / self._a
            modified_wavenumbers[i_sample, :] = np.linalg.eigvals(matrix)
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
        xticks_labels = np.linspace(-degree-1, degree+1, 2*degree+3, dtype=int)
        # xticks_labels = np.linspace(0, degree+1, degree+2, dtype=int)
        xticks_ticks = xticks_labels * np.pi
        kh_min, kh_max = xticks_ticks[0], xticks_ticks[-1]
        sampled_wavenumbers = np.linspace(kh_min, kh_max, n_sample)
        scheme = self.build_scheme(Method, degree, g)
        modified_wavenumbers = self.get_modified_wavenumbers(
            scheme, sampled_wavenumbers)
        physical_eigvals = self.get_physical_mode(sampled_wavenumbers,
            modified_wavenumbers)
        plt.figure(figsize=(6,9))
        plt.subplot(3,1,1)
        plt.ylabel(r'$\Re(\tilde{\kappa}h)$')
        plt.xlabel(r'$\kappa h\,/\,\pi$')
        plt.plot(sampled_wavenumbers, modified_wavenumbers.real, 'k.')
        plt.plot(sampled_wavenumbers, physical_eigvals.real, 'ro', label='Physical')
        plt.plot([kh_min, kh_max], [kh_min, kh_max], '-', label='Exact')
        plt.xticks(xticks_ticks, xticks_labels)
        plt.grid()
        plt.legend()
        plt.subplot(3,1,2)
        plt.ylabel(r'$\Im(\tilde{\kappa}h)$')
        plt.xlabel(r'$\kappa h\,/\,\pi$')
        plt.plot(sampled_wavenumbers, modified_wavenumbers.imag, 'k.')
        plt.plot(sampled_wavenumbers, physical_eigvals.imag, 'ro', label='Physical')
        plt.plot([kh_min, kh_max], [0, 0], '-', label='Exact')
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
        plt.plot([kh_min, 0, kh_max], [-kh_min, 0, kh_max], '-', label='Exact')
        plt.xticks(xticks_ticks, xticks_labels)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        # plt.show()
        scheme = self.build_scheme(Method, degree, g)
        plt.savefig(f'all_modes_of_{scheme.name(False)}_p={degree}.svg')

    def compare_wave_numbers(self, methods, degrees, degree_to_corrections,
            n_sample: int, compressed: bool, name: str):
        linestyles = [
            ('dotted',                (0, (1, 1))),
            ('loosely dotted',        (0, (1, 4))),
            ('dashed',                (0, (4, 4))),
            ('densely dashed',        (0, (4, 1))),
            ('loosely dashed',        (0, (4, 8))),
            ('long dash with offset', (4, (8, 2))),
            ('dashdotted',            (0, (2, 4, 1, 4))),
            ('densely dashdotted',    (0, (2, 1, 1, 1))),
            ('loosely dashdotted',    (0, (2, 8, 1, 8))),
            ('dashdotdotted',         (0, (2, 4, 1, 4, 1, 4))),
            ('densely dashdotdotted', (0, (2, 1, 1, 1, 1, 1))),
            ('loosely dashdotdotted', (0, (2, 8, 1, 8, 1, 8))),]
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
                             label=scheme.name(), linestyle=linestyles[i][1])
                    plt.subplot(2,1,2)
                    plt.plot(sampled_wavenumbers/scale,
                             physical_eigvals.imag/scale,
                             label=scheme.name(), linestyle=linestyles[i][1])
                    i += 1
        x_max = np.max(degrees) * (not compressed) + 1
        plt.subplot(2,1,1)
        plt.plot([0, x_max], [0, x_max], '-', label='Exact')
        plt.grid()
        plt.legend(handlelength=4)
        plt.subplot(2,1,2)
        plt.plot([0, x_max], [0, 0], '-', label='Exact')
        plt.grid()
        plt.legend(handlelength=4)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'{name}.svg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='python3 wave_number.py')
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
    wnd.plot_modified_wavenumbers(SpatialClass, args.degree,
        Vincent(args.degree + 1, Vincent.huynh_lumping_lobatto), args.n_sample)
    wnd.compare_wave_numbers([spatial.LegendreDG,
        spatial.DGonLegendreRoots, spatial.DGonLobattoRoots,
        spatial.FRonLegendreRoots, spatial.FRonUniformRoots,],
        [args.degree], [lambda p: Huynh(p + 1, 2)],
        args.n_sample, args.compressed, 'DGvsFR')
    degree_to_corrections = [
        lambda p: Vincent(p + 1, Vincent.discontinuous_galerkin),
        lambda p: Vincent(p + 1, Vincent.huynh_lumping_lobatto),
        lambda p: Huynh(p + 1, 1),
        lambda p: Huynh(p + 1, 2),
        lambda p: Huynh(p + 1, 3),
        lambda p: Huynh(p + 1, 4),
        lambda p: Huynh(p + 1, 5),
    ]
    wnd.compare_wave_numbers([spatial.FRonLobattoRoots],
        [5], degree_to_corrections,
        args.n_sample, args.compressed, 'HuynhFR')
