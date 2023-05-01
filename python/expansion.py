"""Implement some polynomial approximations for general functions.
"""
import numpy as np
from scipy import special
import numdifftools as nd
from copy import deepcopy

from concept import Expansion, Coordinate
import integrator
import polynomial


class Taylor(Expansion):
    """The Taylor expansion of a general function.

    u^h(x) = \sum_{k=0}^{p} u^{(k)} / (k!) * (x-c)^{k}
    """

    def __init__(self, degree: int, coordinate: Coordinate,
            value_type=float) -> None:
        assert degree >= 0
        self._n_term = degree + 1
        Expansion.__init__(self, coordinate,
            integrator.GaussLegendre(coordinate))
        # coefficients of the Taylor expansion at x_center
        self._taylor_coeff = np.ndarray(self._n_term, value_type)
        self._value_type = value_type

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
        x_center = self.coordinate.x_center()
        self._taylor_coeff[0] = function(x_center)
        for k in range(1, self.n_term()):
            df_dx = nd.Derivative(function, n=k,
                step=self.coordinate.length()/100, order=4)
            derivative = df_dx(x_center)
            # print(derivative)
            self._taylor_coeff[k] = derivative / special.factorial(k)

    def get_average(self):
        def integrand(x_global):
            return self.get_function_value(x_global)
        n_point = 1 + (self.degree() + self.coordinate.jacobian_degree()) // 2
        return self.integrator.average(integrand, n_point)

    def get_basis(self, i_basis: int) -> callable:
        assert 0 <= i_basis
        def function(x_global):
            return (x_global - self.coordinate.x_center())**i_basis
        return function

    def get_basis_values(self, x_global):
        x_global -= self.coordinate.x_center()
        values = x_global**np.arange(0, self.n_term(), dtype=int)
        return values

    def get_basis_gradients(self, x_global: float):
        x_global -= self.coordinate.x_center()
        values = np.ndarray(self.n_term())
        values[0] = 0
        for k in range(self.degree()):
            values[k+1] = (k+1) * x_global**k
        return values

    def get_basis_derivatives(self, x_global: float):
        """Get all non-zero-in-general derivatives of the basis.

        values[k][l] = the k-th derivative of (x-c)^{l}.
        """
        x_global -= self.coordinate.x_center()
        values = np.zeros((self.n_term(), self.n_term()))
        for k in range(1, self.n_term()):
            for l in range(k, self.n_term()):
                values[k][l] = x_global**(l-k) * (special.factorial(l)
                    / special.factorial(l-k))
        return values

    def get_basis_innerproducts(self):
        def integrand(x_global):
            column = self.get_basis_values(x_global)
            matrix = np.tensordot(column, column, axes=0)
            return matrix
        mass_matrix = self.integrator.fixed_quad_global(integrand,
            self.n_term())
        # mass_matrix[i][j] := inner-product of basis[i] and basis[j]
        return mass_matrix

    def get_function_value(self, x_global: float):
        taylor_basis_values = Taylor.get_basis_values(self, x_global)
        return self._taylor_coeff.dot(taylor_basis_values)

    def get_gradient_value(self, x_global: float):
        taylor_basis_gradients = Taylor.get_basis_values(self, x_global)
        return self._taylor_coeff.dot(taylor_basis_gradients)

    def get_derivative_values(self, x_global: float):
        """Get all non-zero-in-general derivatives of u^h.

        values[k] = the k-th derivative of u^h.
        """
        # TODO: evaluate the k-th derivative only
        basis_derivatives = self.get_basis_derivatives(x_global)
        values = np.zeros(self.n_term(), dtype=self._value_type)
        # values[0] = get_function_value(x_global)
        for k in range(1, self.n_term()):
            for l in range(k, self.n_term()):
                values[k] += basis_derivatives[k][l] * self._taylor_coeff[l]
        return values

    def set_coeff(self, coeff):
        self._taylor_coeff[:] = deepcopy(coeff)

    def get_coeff_ref(self):
        return self._taylor_coeff

    def set_taylor_coeff(self, points: np.ndarray):
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

    def convert_to(self, Expansion):
        assert issubclass(Expansion, Taylor)
        that = Expansion(self.degree(), self.coordinate, self._value_type)
        that.approximate(lambda x: self.get_function_value(x))
        return that


class Lagrange(Taylor):
    """The Lagrange expansion of a general function.
    """

    def __init__(self, degree: int, coordinate: Coordinate,
            value_type=float) -> None:
        Taylor.__init__(self, degree, coordinate, value_type)
        n_point = degree + 1
        assert n_point >= 1
        # Sample points evenly distributed in the element.
        # delta = 0.1
        # roots = np.linspace(delta - 1, 1 - delta, n_point)
        # Or, use zeros of special polynomials.
        roots, _ = special.roots_legendre(degree + 1)
        # Build the basis and sample points.
        self._basis = polynomial.LagrangeBasis(roots)
        self._sample_values = np.ndarray(n_point, value_type)
        self._sample_points = self.coordinate.local_to_global(roots)

    def name(self, verbose) -> str:
        my_name = 'Lagrange'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name

    def get_sample_points(self):
        """Get the global coordinates of all sample points."""
        return self._sample_points

    def set_taylor_coeff(self):
        """Transform a Lagrage expansion onto its Taylor basis.

        For Lagrange basis, this_rows will be np.eye(self.n_term()), if
        the i-th row is evaluated at the i-th sample point. So
            base_rows = mat_a => base_col = base_rows^{-1} * this_col.
        """
        base_rows = np.ndarray((self.n_term(), self.n_term()))
        points = self.get_sample_points()
        for k in range(self.n_term()):
            base_rows[k] = Taylor.get_basis_values(self, points[k])
        this_col = self.get_coeff_ref()
        base_col = np.linalg.solve(base_rows, this_col)
        Taylor.set_coeff(self, base_col)

    def set_coeff(self, values):
        """Set values at sample points.

        Users are responsible to ensure (for each i) values[i] is sampled at self._sample_points[i].
        """
        assert len(values) == self._basis.n_term()
        for i in range(len(values)):
            self._sample_values[i] = values[i]
        self.set_taylor_coeff()

    def get_coeff_ref(self):
        return self._sample_values

    def approximate(self, function):
        for i in range(self._basis.n_term()):
            self._sample_values[i] = function(self._sample_points[i])
        self.set_taylor_coeff()

    def get_basis(self, i_basis: int) -> callable:
        assert 0 <= i_basis
        def function(x_global):
            x_local = self.coordinate.global_to_local(x_global)
            return self._basis[i_basis].get_function_value(x_local)
        return function

    def get_basis_values(self, x_global):
        x_local = self.coordinate.global_to_local(x_global)
        values = self._basis.get_function_value(x_local)
        return values

    def get_basis_gradients(self, x_global):
        x_local = self.coordinate.global_to_local(x_global)
        values = self._basis.get_gradient_value(x_local)
        values /= self.coordinate.local_to_jacobian(x_local)
        return values

    def get_function_value(self, x_global):
        values = self.get_basis_values(x_global)
        value = values.dot(self._sample_values)
        return value

    def get_gradient_value(self, x_global):
        basis_grad = self.get_basis_gradients(x_global)
        value = basis_grad.dot(self._sample_values)
        return value


class Legendre(Taylor):
    """Approximate a general function based on Legendre polynomials.
    """

    def __init__(self, degree: int, coordinate: Coordinate,
            value_type=float) -> None:
        Taylor.__init__(self, degree, coordinate, value_type)
        self._mode_coeffs = np.ndarray(self._n_term, value_type)
        # Legendre polynoamials are only orthogonal for 1-degree coordinate map,
        # whose Jacobian determinant is constant over [-1, 1].
        assert self.coordinate.jacobian_degree() == 0
        jacobian = self.coordinate.local_to_jacobian(0)
        # Mode weights are defined as the inner-products of each basis,
        # which can be explicitly integrated here:
        self._mode_weights = jacobian * 2 / (2 * np.arange(degree+1) + 1)
        # taylor_basis_row * matrix_on_taylor = legendre_basis_row
        self._matrix_on_taylor = np.eye(self._n_term)
        for k in range(2, self._n_term):
            self._matrix_on_taylor[1:k+1, k] = ((2*k-1) / k
                * self._matrix_on_taylor[0:k, k-1])
            self._matrix_on_taylor[0:k-1, k] -= ((k-1) / k
                * self._matrix_on_taylor[0:k-1, k-2])
        # Taylor basis is dimensional, but Legendre basis is dimensionless...
        for k in range(1, self._n_term):
            self._matrix_on_taylor[k] /= jacobian**k

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

    def get_average(self):
        return self._mode_coeffs[0]

    def set_taylor_coeff(self):
        """Transform a Lagrage expansion onto its Taylor basis.
        
        For Legendre basis, there is
            taylor_basis_row * matrix_on_taylor = legendre_basis_row
        So
            taylor_coeff_col = matrix_on_taylor * legendre_coeff_col.
        """
        legendre_coeff_col = self.get_coeff_ref()
        taylor_coeff_col = self._matrix_on_taylor.dot(legendre_coeff_col)
        Taylor.set_coeff(self, taylor_coeff_col)

    def set_coeff(self, coeffs):
        """Set coefficient for each mode.
        """
        assert len(coeffs) == self.n_term()
        for i in range(len(coeffs)):
            self._mode_coeffs[i] = coeffs[i]
        self.set_taylor_coeff()

    def get_coeff_ref(self):
        return self._mode_coeffs

    def approximate(self, function):
        for k in range(self.n_term()):
            def integrand(x_local):
                value = special.eval_legendre(k, x_local)
                value *= function(self.coordinate.local_to_global(x_local))
                value *= self.coordinate.local_to_jacobian(x_local)
                return value
            self._mode_coeffs[k] = self.integrator.fixed_quad_local(integrand,
                self.n_term())
            self._mode_coeffs[k] /= self._mode_weights[k]
        self.set_taylor_coeff()

    def get_basis(self, i_basis: int) -> callable:
        assert 0 <= i_basis
        def function(x_global):
            x_local = self.coordinate.global_to_local(x_global)
            return special.eval_legendre(i_basis, x_local)
        return function

    def get_basis_values(self, x_global):
        x_local = self.coordinate.global_to_local(x_global)
        values = np.ndarray(self.n_term())
        for k in range(self.n_term()):
            values[k] = special.eval_legendre(k, x_local)
        return values

    def get_basis_gradients(self, x_global):
        x_local = self.coordinate.global_to_local(x_global)
        values = np.ndarray(self.n_term())
        values[0] = 0.0
        for k in range(1, self.n_term()):
            values[k] = (k * special.eval_legendre(k-1, x_local)
                + x_local * values[k-1])
        values /= self.coordinate.global_to_jacobian(x_global)
        return values

    def get_function_value(self, x_global):
        values = self.get_basis_values(x_global)
        value = values.dot(self._mode_coeffs)
        return value

    def get_gradient_value(self, x_global):
        basis_grad = self.get_basis_gradients(x_global)
        value = basis_grad.dot(self._mode_coeffs)
        return value


class TruncatedLegendre(Taylor):
    """A lower-order view of a higher-order Legendre expansion.
    """

    def __init__(self, degree: int, that: Legendre) -> None:
        assert 0 <= degree <= that.degree()
        assert isinstance(that, Legendre)
        Taylor.__init__(self, degree, that.coordinate, that._value_type)
        n_term = degree + 1
        self._taylor_coeff[:] = deepcopy(that._taylor_coeff[0:n_term])
        assert isinstance(that, Legendre) # TODO: relax to Taylor
        self._mode_coeffs = deepcopy(that._mode_coeffs[0:n_term])
        self._matrix_on_taylor = deepcopy(
            that._matrix_on_taylor[0:n_term, 0:n_term])
        Legendre.set_taylor_coeff(self)

    def name(self, verbose) -> str:
        my_name = 'TruncatedLegendre'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name

    def set_coeff(self):
        assert False

    def get_coeff_ref(self):
        return Legendre.get_coeff_ref(self)

    def get_average(self):
        return Legendre.get_average(self)

    def get_basis(self, i_basis: int) -> callable:
        return Legendre.get_basis(self, i_basis)

    def get_basis_values(self, x_global):
        return Legendre.get_basis_values(self, x_global)

    def get_basis_gradients(self, x_global: float):
        assert False


if __name__ == '__main__':
    pass
