"""Concrete implementations of spatial schemes.
"""
import numpy as np
from numpy.testing import assert_almost_equal
import bisect

import concept
from  coordinate import LinearCoordinate
import element


class FiniteElement(concept.SpatialScheme):
    """The base of all finite element schemes for conservation laws.
    """

    def __init__(self, riemann: concept.RiemannSolver,
            degree: int, n_element: int, x_left: float, x_right: float,
            ElementType: concept.Element) -> None:
        concept.SpatialScheme.__init__(self, riemann,
            n_element, x_left, x_right)
        assert degree >= 0
        delta_x = (x_right - x_left) / n_element
        self._x_left_sorted = np.ndarray(n_element)
        x_left_i = x_left
        for i_element in range(n_element):
            assert_almost_equal(x_left_i, x_left + i_element * delta_x)
            self._x_left_sorted[i_element] = x_left_i
            x_right_i = x_left_i + delta_x
            element_i = ElementType(riemann, degree,
                LinearCoordinate(x_left_i, x_right_i))
            self._elements[i_element] = element_i
            x_left_i = x_right_i
        assert_almost_equal(x_left_i, x_right)
        self.link_neighbors()

    def n_dof(self):
        return self.n_element() * self.get_element_by_index(0).n_dof()

    def get_element_index(self, x_global):
        i_element = bisect.bisect_right(self._x_left_sorted, x_global)
        # bisect_right(a, x) gives such an i that a[:i] <= x < a[i:]
        return i_element - 1

    def get_interface_fluxes(self):
        """Get the interface flux at each element interface.
        """
        interface_fluxes = np.ndarray(self.n_element() + 1, self.value_type())
        # interface_flux[i] := flux on interface(element[i-1], element[i])
        for i in range(1, self.n_element()):
            curr = self.get_element_by_index(i)
            prev = self.get_element_by_index(i-1)
            viscous = self.equation().get_diffusive_coeff()
            if self._viscous:
                nu_curr = self._viscous.get_coeff(i)
                if callable(nu_curr):
                    nu_curr = nu_curr(curr.x_left())
                nu_prev = self._viscous.get_coeff(i-1)
                if callable(nu_prev):
                    nu_prev = nu_prev(prev.x_right())
                viscous += min(nu_curr, nu_prev)
            interface_fluxes[i] = self._riemann.get_interface_flux(
                prev.expansion(), curr.expansion(), viscous)
        if self.is_periodic():
            i_prev = self.n_element() - 1
            curr = self.get_element_by_index(0)
            prev = self.get_element_by_index(i_prev)
            viscous = self.equation().get_diffusive_coeff()
            if self._viscous:
                nu_curr = self._viscous.get_coeff(0)
                if callable(nu_curr):
                    nu_curr = nu_curr(curr.x_left())
                nu_prev = self._viscous.get_coeff(i_prev)
                if callable(nu_prev):
                    nu_prev = nu_prev(prev.x_right())
                viscous += min(nu_curr, nu_prev)
            interface_fluxes[0] = self._riemann.get_interface_flux(
                prev.expansion(), curr.expansion(), viscous)
            interface_fluxes[-1] = interface_fluxes[0]
        else:  # TODO: support other boundary condtions
            curr = self.get_element_by_index(0)
            viscous = 0.0
            if self._viscous:
                viscous = self._viscous.get_coeff(0)
            interface_fluxes[0] = \
                curr.get_discontinuous_flux(curr.x_left(), viscous)
            curr = self.get_element_by_index(-1)
            viscous = 0.0
            if self._viscous:
                viscous = self._viscous.get_coeff(-1)
            interface_fluxes[-1] = \
                curr.get_discontinuous_flux(curr.x_right(), viscous)
        return interface_fluxes

    def get_solution_value(self, point):
        return self.get_element(point).get_solution_value(point)

    def get_discontinuous_flux(self, point):
        i_cell = self.get_element_index(point)
        cell_i = self.get_element_by_index(i_cell)
        viscous = 0.0
        if self._viscous:
            viscous = self._viscous.get_coeff(i_cell)
        return cell_i.get_discontinuous_flux(point, viscous)

    def set_solution_column(self, column):
        assert len(column) == self.n_dof()
        first = 0
        for element_i in self._elements:
            assert isinstance(element_i, concept.Element)
            last = first + element_i.n_dof()
            element_i.set_solution_coeff(column[first:last])
            first = last
        assert first == self.n_dof()
        self.suppress_oscillations()

    def get_solution_column(self):
        column = np.zeros(self.n_dof(), self.scalar_type())
        first = 0
        for element_i in self._elements:
            assert isinstance(element_i, concept.Element)
            last = first + element_i.n_dof()
            column[first:last] = element_i.get_solution_column()
            first = last
        assert first == self.n_dof()
        return column

    def _write_to_column(self, column: np.ndarray, values: np.ndarray, i_dof):
        n_row = len(values)
        if np.isscalar(values[0]):
            column[i_dof:i_dof+n_row] = values
            i_dof += n_row
        else:
            n_col = len(values[0])
            for i_col in range(n_col):
                for i_row in range(n_row):
                    column[i_dof] = values[i_row][i_col]
                    i_dof += 1
        return i_dof

    def initialize(self, function: callable):
        for element_i in self._elements:
            assert isinstance(element_i, concept.Element)
            element_i.approximate(function)
        self.suppress_oscillations()


class DiscontinuousGalerkin(FiniteElement):
    """An mid-level class that defines common methods for all DG schemes.
    """

    def get_residual_column(self):
        column = np.zeros(self.n_dof(), self.scalar_type())
        interface_fluxes = self.get_interface_fluxes()
        i_dof = 0
        for i in range(self.n_element()):
            element_i = self.get_element_by_index(i)
            # build element_i's residual column
            # 1st: evaluate the internal integral
            extra_viscous = 0.0
            if self._viscous:
                extra_viscous = self._viscous.get_coeff(i)
            def integrand(x_global):
                return np.tensordot(
                    element_i.get_basis_gradients(x_global),
                    element_i.get_discontinuous_flux(x_global, extra_viscous),
                    0)
            values = element_i.fixed_quad_global(integrand, element_i.degree())
            # 2nd: evaluate the boundary integral
            upwind_flux_left = interface_fluxes[i]
            upwind_flux_right = interface_fluxes[i+1]
            values += np.tensordot(
                element_i.get_basis_values(element_i.x_left()),
                upwind_flux_left, 0)
            values -= np.tensordot(
                element_i.get_basis_values(element_i.x_right()),
                upwind_flux_right, 0)
            # 3rd: multiply the inverse of the mass matrix
            values = element_i.divide_mass_matrix(values)
            # write to the global column
            i_dof = self._write_to_column(column, values, i_dof)
        assert i_dof == self.n_dof()
        return column

    def get_flux_value(self, point):
        return self.get_discontinuous_flux(point)


class LegendreDG(DiscontinuousGalerkin):
    """The ODE system given by the DG method using a Legendre expansion.
    """

    def __init__(self, riemann: concept.RiemannSolver,
            degree: int, n_element: int, x_left: float, x_right: float) -> None:
        FiniteElement.__init__(self, riemann, degree,
            n_element, x_left, x_right, element.LegendreDG)

    def name(self, verbose=True):
        my_name = 'LegendreDG'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name


class LagrangeDG(DiscontinuousGalerkin):
    """The ODE system given by the DG method using a Lagrange expansion.
    """

    def __init__(self, riemann: concept.RiemannSolver,
            degree: int, n_element: int, x_left: float, x_right: float) -> None:
        DiscontinuousGalerkin.__init__(self, riemann, degree,
            n_element, x_left, x_right, element.LagrangeDG)

    def name(self, verbose=True):
        my_name = 'LagrangeDG'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name


class FluxReconstruction(FiniteElement):
    """An mid-level class that defines common methods for all FR schemes.
    """

    def get_residual_column(self):
        column = np.zeros(self.n_dof(), self.scalar_type())
        interface_fluxes = self.get_interface_fluxes()
        # evaluate flux gradients
        i_dof = 0
        for i in range(self.n_element()):
            element_i = self.get_element_by_index(i)
            assert (isinstance(element_i, element.LagrangeFR)
                or isinstance(element_i, element.LegendreFR))
            extra_viscous = 0.0
            if self._viscous:
                extra_viscous = self._viscous.get_coeff(i)
            upwind_flux_left = interface_fluxes[i]
            upwind_flux_right = interface_fluxes[i+1]
            values = -element_i.get_flux_gradients(
                upwind_flux_left, upwind_flux_right, extra_viscous)
            i_dof = self._write_to_column(column, values, i_dof)
        assert i_dof == self.n_dof(), (i_dof, self.n_dof())
        return column

    def get_continuous_flux(self, point):
        curr = self.get_element_index(point)
        viscous = self.equation().get_diffusive_coeff()
        # solve riemann problem at the left end of curr element
        right = self.get_element_by_index(curr)
        left = self.get_element_by_index(curr-1)
        upwind_flux_left = self._riemann.get_interface_flux(
            left.expansion(), right.expansion(), viscous)
        # solve riemann problem at the right end of curr element
        left = self.get_element_by_index(curr)
        right = self.get_element_by_index((curr + 1) % self.n_element())
        upwind_flux_right = self._riemann.get_interface_flux(
            left.expansion(), right.expansion(), viscous)
        assert (isinstance(left, element.LagrangeFR)
            or isinstance(left, element.LegendreFR))
        return left.get_continuous_flux(point,
            upwind_flux_left, upwind_flux_right)

    def get_flux_value(self, point):
        return self.get_continuous_flux(point)


class LagrangeFR(FluxReconstruction):
    """The ODE system given by Huyhn's FR method.
    """

    def __init__(self, riemann: concept.RiemannSolver,
            degree: int, n_element: int, x_left: float, x_right: float) -> None:
        FluxReconstruction.__init__(self, riemann, degree,
            n_element, x_left, x_right, element.LagrangeFR)

    def name(self, verbose=True):
        my_name = 'LagrangeFR'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name


class LegendreFR(FluxReconstruction):
    """The ODE system given by Huyhn's FR method.
    """

    def __init__(self, riemann: concept.RiemannSolver,
            degree: int, n_element: int, x_left: float, x_right: float) -> None:
        FluxReconstruction.__init__(self, riemann, degree,
            n_element, x_left, x_right, element.LegendreFR)

    def name(self, verbose=True):
        my_name = 'LegendreFR'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name


if __name__ == '__main__':
    pass
