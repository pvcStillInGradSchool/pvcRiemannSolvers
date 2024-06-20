"""Implement some polynomial approximations for general functions.
"""
import numpy as np
from scipy import special
import numdifftools as nd

import concept
import coordinate
import integrator
import polynomial


class Shifted(concept.Expansion):
    """An wrapper that acts as if a given expansion is shifted along x-axis by a given amount.
    """

    def __init__(self, expansion: concept.Expansion, x_shift: float):
        shifted_coordinate = coordinate.Shifted(expansion.coordinate(), x_shift)
        IntegtatorType = type(expansion.integrator())
        n_point = expansion.integrator().n_point()
        concept.Expansion.__init__(self,
            IntegtatorType(shifted_coordinate, n_point), expansion.value_type())
        self._unshifted_expansion = expansion
        self._x_shift = x_shift

    def scalar_type(self):
        return self._unshifted_expansion.scalar_type()

    def name(self, verbose: bool) -> str:
        return self._unshifted_expansion.name(verbose)

    def n_term(self):
        return self._unshifted_expansion.n_term()

    def degree(self):
        return self._unshifted_expansion.degree()

    def global_to_value(self, x_global: float):
        x_unshifted = x_global - self._x_shift
        return self._unshifted_expansion.global_to_value(x_unshifted)

    def global_to_gradient(self, x_global: float):
        x_unshifted = x_global - self._x_shift
        return self._unshifted_expansion.global_to_gradient(x_unshifted)

    def global_to_derivatives(self, x_global: float, k: int) -> np.ndarray:
        x_unshifted = x_global - self._x_shift
        return self._unshifted_expansion.global_to_derivatives(x_unshifted, k)

    def get_basis_values(self, x_global: float) -> np.ndarray:
        x_unshifted = x_global - self._x_shift
        return self._unshifted_expansion.get_basis_values(x_unshifted)

    def get_basis_gradients(self, x_global: float) -> np.ndarray:
        x_unshifted = x_global - self._x_shift
        return self._unshifted_expansion.get_basis_gradients(x_unshifted)

    def get_basis_hessians(self, x_global: float) -> np.ndarray:
        x_unshifted = x_global - self._x_shift
        return self._unshifted_expansion.get_basis_hessians(x_unshifted)

    def get_basis_derivatives(self, x_global: float, k: int) -> np.ndarray:
        x_unshifted = x_global - self._x_shift
        return self._unshifted_expansion.get_basis_derivatives(x_unshifted, k)

    def get_boundary_basis_derivatives(self, k: int, left: bool):
        return self._unshifted_expansion.get_boundary_basis_derivatives(k, left)

    def get_coeff_ref(self) -> np.ndarray:
        return self._unshifted_expansion.get_coeff_ref()

    def average(self):
        return self._unshifted_expansion.average()

    def get_basis_innerproducts(self):
        return self._unshifted_expansion.get_basis_innerproducts()

    # The following methods are banned for this wrapper.

    def approximate(self, function: callable):
        assert False

    def get_basis(self, i_basis: int) -> callable:
        assert False

    def set_coeff(self, coeff: np.ndarray):
        assert False


class Taylor(concept.Expansion):
    """The Taylor expansion of a general function.

    \f$ u^h(x) = \sum_{k=0}^{p} \frac{ u^{(k)}(c) }{ k! } (x-c)^{k} \f$
    """

    _factorials = np.ones(100)
    for k in range(1, len(_factorials)):
        _factorials[k] = _factorials[k-1] * k

    @staticmethod
    def get_integrator(degree: int, coordinate: concept.Coordinate,
            IntegratorType = integrator.GaussLegendre) -> concept.Integrator:
        return IntegratorType(coordinate, degree + 1)

    def __init__(self, degree: int, integrator: concept.Integrator,
            value_type=float) -> None:
        assert 0 <= degree < 100
        self._n_term = degree + 1
        concept.Expansion.__init__(self, integrator, value_type)
        # coefficients of the Taylor expansion at x_center
        self._taylor_coeff = np.ndarray(self._n_term, value_type)

    def scalar_type(self):
        value = self._taylor_coeff[0]
        if np.isscalar(value):
            return type(value)
        else:
            return type(value[0])

    def name(self, verbose) -> str:
        my_name = 'Taylor'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name

    def n_term(self):
        return self._n_term

    def degree(self):
        return self.n_term() - 1

    def approximate(self, function: callable):
        x_center = self.x_center()
        self._taylor_coeff[0] = function(x_center)
        def kth_df_dx(f, k) -> callable:
            return nd.Derivative(f, n=k, step=self.length()/100, order=4)
        for k in range(1, self.n_term()):
            if issubclass(self.scalar_type(), complex):
                def df_dx(x):
                    real = kth_df_dx(lambda x: function(x).real, k)
                    imag = kth_df_dx(lambda x: function(x).imag, k)
                    return real(x) + 1j * imag(x)
            elif issubclass(self.scalar_type(), float):
                df_dx = kth_df_dx(function, k)
            else:
                assert False
            derivative = df_dx(x_center)
            # print(derivative)
            self._taylor_coeff[k] = derivative / Taylor._factorials[k]

    def average(self):
        def integrand(x_global):
            return self.global_to_value(x_global)
        return self.integrator().average(integrand)

    def get_basis(self, i_basis: int) -> callable:
        assert 0 <= i_basis
        def function(x_global):
            return (x_global - self.x_center())**i_basis
        return function

    def get_basis_values(self, x_global):
        x_global -= self.x_center()
        values = np.ndarray(self.n_term())
        x_power = 1.0
        values[0] = 1.0
        for k in range(0, self.n_term()):
            values[k] = x_power
            x_power *= x_global
        return values

    def get_basis_gradients(self, x_global: float):
        x_global -= self.x_center()
        values = np.ndarray(self.n_term())
        x_power = 1.0
        values[0] = 0
        for k in range(1, self.n_term()):
            values[k] = k * x_power
            x_power *= x_global
        return values

    def get_basis_hessians(self, x_global: float):
        x_global -= self.x_center()
        values = np.ndarray(self.n_term())
        x_power = 1.0
        values[0] = 0
        if self.degree() > 0:
            values[1] = 0
        for k in range(2, self.n_term()):
            values[k] = k * (k-1) * x_power
            x_power *= x_global
        return values

    def get_basis_derivatives(self, x_global: float, k: int):
        if k == 0:
            return Taylor.get_basis_values(self, x_global)
        else:
            assert k > 0
            values = np.zeros(self.n_term())
            x_global -= self.x_center()
            for l in range(k, self.n_term()):
                values[l] = x_global**(l-k) * (Taylor._factorials[l]
                    / Taylor._factorials[l-k])
            return values

    def get_basis_innerproducts(self):
        def integrand(x_global):
            column = self.get_basis_values(x_global)
            matrix = np.tensordot(column, column, axes=0)
            return matrix
        mass_matrix = self.integrator().integrate(integrand)
        # mass_matrix[i][j] := inner-product of basis[i] and basis[j]
        return mass_matrix

    def global_to_value(self, x_global: float):
        taylor_basis_values = Taylor.get_basis_values(self, x_global)
        return self._taylor_coeff.dot(taylor_basis_values)

    def global_to_gradient(self, x_global: float):
        taylor_basis_gradients = Taylor.get_basis_gradients(self, x_global)
        return self._taylor_coeff.dot(taylor_basis_gradients)

    def global_to_derivatives(self, x_global: float, k: int) -> np.ndarray:
        if k == 0:
            return self.global_to_value(x_global)
        assert k > 0
        taylor_derivatives = Taylor.get_basis_derivatives(self, x_global, k)
        value = taylor_derivatives[k] * self._taylor_coeff[k]
        for l in range(k + 1, self.n_term()):
            value += taylor_derivatives[l] * self._taylor_coeff[l]
        return value

    def set_coeff(self, coeff):
        for i_row in range(self.n_term()):
            self._taylor_coeff[i_row] = coeff[i_row]

    def get_coeff_ref(self):
        return self._taylor_coeff

    def _set_taylor_coeff(self, points: np.ndarray):
        """Transform another polynomial expansion onto its Taylor basis.

        Store basis values in a row, and coefficients in a column, one has
            base_row = this_row * mat_a => base_col = mat_a^{-1} * this_col.
        """
        assert issubclass(type(self), Taylor)
        assert len(points) == self.n_term()
        this_rows = np.ndarray((self.n_term(), self.n_term()))
        base_rows = np.ndarray((self.n_term(), self.n_term()))
        for k in range(self.n_term()):
            this_rows[k] = self.get_basis_values(points[k])
            base_rows[k] = Taylor.get_basis_values(self, points[k])
        mat_a = np.linalg.solve(this_rows, base_rows)
        this_col = self.get_coeff_ref()
        base_col = np.linalg.solve(mat_a, this_col)
        Taylor.set_coeff(self, base_col)

    def set_taylor_coeff(self, taylor_coeff: np.ndarray):
        """Update the Expansion by setting its Taylor coeff.
        """
        Taylor.set_coeff(self, taylor_coeff)

    def convert_to(self, Expansion):
        assert issubclass(Expansion, Taylor)
        that = Expansion(self.degree(), self.coordinate(), self.value_type())
        that.approximate(lambda x: self.global_to_value(x))
        return that


class Lagrange(Taylor):
    """The Lagrange expansion of a general function.
    """

    def __init__(self, degree: int, integrator: concept.Integrator,
            get_local_points: callable, value_type=float) -> None:
        Taylor.__init__(self, degree, integrator, value_type)
        n_point = degree + 1
        assert n_point >= 1
        roots = get_local_points(n_point)
        # Build the basis and sample points.
        self._basis = polynomial.LagrangeBasis(roots)
        self._sample_values = np.ndarray(n_point, value_type)
        Integrator = type(integrator)
        if get_local_points is Integrator.get_local_points:
            self._sample_points = self.integrator().get_global_points()
        else:
            self._sample_points = self.coordinate().local_to_global(roots)
        # base_rows[k] := values of the base (Taylor) basis at sample_points[k]
        base_rows = np.ndarray((self.n_term(), self.n_term()))
        for k in range(self.n_term()):
            x_global = self._sample_points[k]
            base_rows[k] = Taylor.get_basis_values(self, x_global)
        """
        Since
            base_rows * lagrange_to_taylor = lagrange_basis_rows = eye,
        we have
            lagrange_to_taylor = base_rows^{-1},
        which can be used in
            taylor_coeff_col = lagrange_to_taylor * lagrange_coeff_col.
        """
        self._taylor_to_lagrange = base_rows
        self._lagrange_to_taylor = np.linalg.inv(base_rows)
        self._cached_average = None
        # cached for frequent usage:
        self._basis_derivatives = np.zeros((n_point, n_point, n_point))
        for i_point in range(n_point):
            point_i = self._sample_points[i_point]
            for k_derivative in range(n_point):
                self._basis_derivatives[i_point][k_derivative] = \
                    self.get_basis_derivatives(point_i, k_derivative)

    def _update_cached_average(self):
        self._cached_average = Taylor.average(self)

    def average(self):
        return self._cached_average

    def name(self, verbose) -> str:
        my_name = 'Lagrange'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name

    def get_sample_points(self) -> np.ndarray:
        """Get the global coordinates of all sample points."""
        return self._sample_points

    def get_sample_values(self) -> np.ndarray:
        """Get the sampled values on each sample point."""
        return self.get_coeff_ref()

    def _set_taylor_coeff(self):
        """Transform a Lagrage expansion onto its Taylor basis.
        """
        lagrange_col = self.get_coeff_ref()
        taylor_col = self._lagrange_to_taylor.dot(lagrange_col)
        Taylor.set_coeff(self, taylor_col)

    def set_taylor_coeff(self, taylor_coeff: np.ndarray):
        Taylor.set_coeff(self, taylor_coeff)
        lagrange_coeff = self._taylor_to_lagrange @ taylor_coeff
        for i in range(len(lagrange_coeff)):
            self._sample_values[i] = lagrange_coeff[i]
        self._update_cached_average()

    def set_coeff(self, values):
        """Set values at sample points.

        Users are responsible to ensure (for each i) values[i] is sampled at self._sample_points[i].
        """
        assert len(values) == self._basis.n_term()
        for i in range(len(values)):
            self._sample_values[i] = values[i]
        self._set_taylor_coeff()
        self._update_cached_average()

    def get_coeff_ref(self):
        return self._sample_values

    def approximate(self, function):
        for i in range(self._basis.n_term()):
            self._sample_values[i] = function(self._sample_points[i])
        self._set_taylor_coeff()
        self._update_cached_average()

    def get_basis(self, i_basis: int) -> callable:
        assert 0 <= i_basis
        def function(x_global):
            x_local = self.coordinate().global_to_local(x_global)
            return self._basis[i_basis].local_to_value(x_local)
        return function

    def get_basis_values(self, x_global):
        x_local = self.coordinate().global_to_local(x_global)
        values = self._basis.local_to_value(x_local)
        return values

    def get_basis_gradients(self, x_global):
        """
        Since
            taylor_basis_row * lagrange_to_taylor = lagrange_basis_row,
        we have
            taylor_gradient_row * lagrange_to_taylor = lagrange_gradient_row.
        """
        taylor_gradients = Taylor.get_basis_gradients(self, x_global)
        return taylor_gradients.dot(self._lagrange_to_taylor)

    def get_basis_hessians(self, x_global: float):
        taylor_hessians = Taylor.get_basis_hessians(self, x_global)
        return taylor_hessians.dot(self._lagrange_to_taylor)

    def get_basis_derivatives(self, x_global: float, k: int):
        taylor_derivatives = Taylor.get_basis_derivatives(self, x_global, k)
        return taylor_derivatives.dot(self._lagrange_to_taylor)

    def get_derivatives_at_node(self, j_node, k_order, basis: bool):
        if k_order == 0:
            if basis:
                return self._basis_derivatives[j_node][k_order]
            else:
                return self._sample_values[j_node]
        elif k_order > self.degree():
            basis_derivatives = np.zeros(self.n_term())
        else:
            basis_derivatives = self._basis_derivatives[j_node][k_order]
        if basis:
            return basis_derivatives
        else:
            return self._sample_values.dot(basis_derivatives)


class LagrangeOnUniformRoots(Lagrange):
    """Specialized Lagrange expansion using uniform roots as nodes.
    """

    @staticmethod
    def _get_local_roots(n_point):
        # Sample points evenly distributed in the element.
        delta = 2.0 / n_point / 2
        roots = np.linspace(delta - 1, 1 - delta, n_point)
        return roots

    def __init__(self, degree: int, coordinate: concept.Coordinate,
            value_type=float) -> None:
        integrator = Taylor.get_integrator(degree, coordinate)
        get_local_points = LagrangeOnUniformRoots._get_local_roots
        Lagrange.__init__(self, degree, integrator,
            get_local_points, value_type)

    def name(self, verbose) -> str:
        my_name = 'LagrangeOnUniformRoots'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name


class LagrangeOnGaussPoints(Lagrange):
    """Specialized Lagrange expansion using Gauss points as nodes.
    """

    def __init__(self, degree: int, coordinate: concept.Coordinate,
            Integrator, value_type=float) -> None:
        assert issubclass(Integrator, concept.Integrator)
        integrator = Taylor.get_integrator(degree, coordinate, Integrator)
        Lagrange.__init__(self, degree, integrator,
            Integrator.get_local_points, value_type)
        assert self._sample_points is self.integrator().get_global_points()
        self._sample_weights = self.integrator().get_global_weights()

    def name(self, verbose) -> str:
        my_name = 'LagrangeOnGaussPoints'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name

    def get_sample_weight(self, k: int) -> float:
        """Get the gloabl quadrature weight of the kth node.
        """
        return self._sample_weights[k]

    def _update_cached_average(self):
        values = self.get_sample_values()
        value = values[0] * self.get_sample_weight(0)
        for i in range(1, self.n_term()):
            value += values[i] * self.get_sample_weight(i)
        self._cached_average = value / self.coordinate().length()


class LagrangeOnLegendreRoots(LagrangeOnGaussPoints):
    """Specialized Lagrange expansion using Legendre roots as nodes.
    """

    def __init__(self, degree: int, coordinate: concept.Coordinate,
            value_type=float) -> None:
        LagrangeOnGaussPoints.__init__(self, degree, coordinate,
            integrator.GaussLegendre, value_type)
        shape = (self.n_term(), self.n_term())
        self._basis_derivatives_at_left = np.ndarray(shape)
        self._basis_derivatives_at_right = np.ndarray(shape)
        for k in range(self.n_term()):
            self._basis_derivatives_at_left[k] = \
                self.get_basis_derivatives(self.x_left(), k)
            self._basis_derivatives_at_right[k] = \
                self.get_basis_derivatives(self.x_right(), k)

    def name(self, verbose) -> str:
        my_name = 'LagrangeOnLegendreRoots'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name

    def get_boundary_basis_derivatives(self, k: int, left: bool):
        if k > self.degree():
            return np.zeros(self.n_term())
        if left:
            return self._basis_derivatives_at_left[k]
        else:
            return self._basis_derivatives_at_right[k]


class LagrangeOnLobattoRoots(LagrangeOnGaussPoints):
    """Specialized Lagrange expansion using Lobatto roots as nodes.
    """

    def __init__(self, degree: int, coordinate: concept.Coordinate,
            value_type=float) -> None:
        LagrangeOnGaussPoints.__init__(self, degree, coordinate,
            integrator.GaussLobatto, value_type)

    def name(self, verbose) -> str:
        my_name = 'LagrangeOnLobattoRoots'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name

    def get_boundary_derivatives(self, k_order: int, left: bool, basis: bool):
        if left:
            j_node = 0
        else:
            j_node = -1
        return self.get_derivatives_at_node(j_node, k_order, basis)

    def approximate(self, function):
        self._sample_values[0] = function(self._sample_points[0] + 1e-10)
        for i in range(1, self._basis.n_term() - 1):
            self._sample_values[i] = function(self._sample_points[i])
        self._sample_values[-1] = function(self._sample_points[-1] - 1e-10)
        self._set_taylor_coeff()
        self._update_cached_average()


class Legendre(Taylor):
    """Approximate a general function based on Legendre polynomials.
    """

    def __init__(self, degree: int, coordinate: concept.Coordinate,
            value_type=float) -> None:
        integrator = Taylor.get_integrator(degree, coordinate)
        Taylor.__init__(self, degree, integrator, value_type)
        self._mode_coeffs = np.ndarray(self._n_term, value_type)
        # Legendre polynoamials are only orthogonal for 1-degree coordinate map,
        # whose Jacobian determinant is constant over [-1, 1].
        assert self.coordinate().jacobian_degree() == 0
        jacobian = self.coordinate().local_to_jacobian(0)
        # Mode weights are defined as the inner-products of each basis,
        # which can be explicitly integrated here:
        self._mode_weights = jacobian * 2 / (2 * np.arange(degree+1) + 1)
        # taylor_basis_row * _legendre_to_taylor = legendre_basis_row
        self._legendre_to_taylor = np.eye(self._n_term)
        for k in range(2, self._n_term):
            self._legendre_to_taylor[1:k+1, k] = ((2*k-1) / k
                * self._legendre_to_taylor[0:k, k-1])
            self._legendre_to_taylor[0:k-1, k] -= ((k-1) / k
                * self._legendre_to_taylor[0:k-1, k-2])
        # Taylor basis is dimensional, but Legendre basis is dimensionless...
        for k in range(1, self._n_term):
            self._legendre_to_taylor[k] /= jacobian**k
        self._taylor_to_legendre = np.linalg.inv(self._legendre_to_taylor)

    def name(self, verbose) -> str:
        my_name = 'Legendre'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name

    def get_mode_weight(self, k):
        """Get the inner-product of the kth basis with itself.
        """
        return self._mode_weights[k]

    def get_mode_energy(self, k):
        """Get the inner-product of the kth component with itself.
        """
        return self._mode_weights[k] * self._mode_coeffs[k]**2

    def average(self):
        return self._mode_coeffs[0]

    def _set_taylor_coeff(self):
        """Transform a Lagrage expansion onto its Taylor basis.
        
        For Legendre basis, there is
            taylor_basis_row * legendre_to_taylor = legendre_basis_row
        So
            taylor_coeff_col = legendre_to_taylor * legendre_coeff_col.
        """
        legendre_coeff_col = self.get_coeff_ref()
        taylor_coeff_col = self._legendre_to_taylor.dot(legendre_coeff_col)
        Taylor.set_coeff(self, taylor_coeff_col)

    def set_taylor_coeff(self, taylor_coeff: np.ndarray):
        Taylor.set_coeff(self, taylor_coeff)
        legendre_coeff = self._taylor_to_legendre @ taylor_coeff
        for i in range(len(legendre_coeff)):
            self._mode_coeffs[i] = legendre_coeff[i]

    def set_coeff(self, coeffs):
        """Set coefficient for each mode.
        """
        assert len(coeffs) == self.n_term()
        for i in range(len(coeffs)):
            self._mode_coeffs[i] = coeffs[i]
        self._set_taylor_coeff()

    def get_coeff_ref(self):
        return self._mode_coeffs

    def approximate(self, function):
        for k in range(self.n_term()):
            def integrand(x_global):
                x_local = self.coordinate().global_to_local(x_global)
                value = special.eval_legendre(k, x_local) * function(x_global)
                return value
            self._mode_coeffs[k] = self.integrator().integrate(integrand)
            self._mode_coeffs[k] /= self._mode_weights[k]
        self._set_taylor_coeff()

    def get_basis(self, i_basis: int) -> callable:
        assert 0 <= i_basis
        def function(x_global):
            x_local = self.coordinate().global_to_local(x_global)
            return special.eval_legendre(i_basis, x_local)
        return function

    def get_basis_values(self, x_global):
        x_local = self.coordinate().global_to_local(x_global)
        values = np.ndarray(self.n_term())
        for k in range(self.n_term()):
            values[k] = special.eval_legendre(k, x_local)
        return values

    def get_basis_gradients(self, x_global):
        taylor_gradients = Taylor.get_basis_gradients(self, x_global)
        return taylor_gradients.dot(self._legendre_to_taylor)

    def get_basis_hessians(self, x_global: float):
        taylor_hessians = Taylor.get_basis_hessians(self, x_global)
        return taylor_hessians.dot(self._legendre_to_taylor)

    def get_basis_derivatives(self, x_global: float, k: int):
        taylor_derivatives = Taylor.get_basis_derivatives(self, x_global, k)
        return taylor_derivatives.dot(self._legendre_to_taylor)


class TruncatedLegendre(Taylor):
    """A lower-order view of a higher-order Legendre expansion.
    """

    def __init__(self, degree: int, that: Legendre) -> None:
        assert 0 <= degree <= that.degree()
        assert isinstance(that, Legendre)
        Taylor.__init__(self, degree, that.integrator(), that.value_type())
        n_term = degree + 1
        self._taylor_coeff[:] = that._taylor_coeff[0:n_term]
        self._mode_coeffs = that._mode_coeffs[0:n_term]
        self._legendre_to_taylor = \
            that._legendre_to_taylor[0:n_term, 0:n_term]
        Legendre._set_taylor_coeff(self)

    def name(self, verbose) -> str:
        my_name = 'TruncatedLegendre'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name

    def set_coeff(self, coeff):
        assert False

    def set_taylor_coeff(self, taylor_coeff):
        assert False

    def get_coeff_ref(self):
        return Legendre.get_coeff_ref(self)

    def average(self):
        return Legendre.average(self)

    def get_basis(self, i_basis: int) -> callable:
        return Legendre.get_basis(self, i_basis)

    def get_basis_values(self, x_global):
        return Legendre.get_basis_values(self, x_global)

    def get_basis_gradients(self, x_global: float):
        assert False

    def get_basis_hessians(self, x_global: float):
        assert False

    def get_basis_derivatives(self, x_global: float, k: int):
        assert False


if __name__ == '__main__':
    pass
