"""Define the abstract interfaces that must be defined in concrete implementations.
"""
import abc
import numpy as np


class Polynomial(abc.ABC):
    """Polynomial functions defined on [-1, 1].
    """
    # TODO: rename to LocalPolynomial

    @abc.abstractmethod
    def name(self, verbose: bool) -> str:
        """Get the name of the polynomial.
        """

    @abc.abstractmethod
    def degree(self):
        """Degree of the polynomial.
        """

    def p_str(self) -> str:
        p = r'$p$'
        return f'{p}={self.degree()}'

    @abc.abstractmethod
    def local_to_value(self, x_local):
        """Evaluate the polynomial and return-by-value the result.
        """

    @abc.abstractmethod
    def local_to_gradient(self, x_local):
        """Evaluate the gradient of the polynomial and return-by-value the result.
        """


class Coordinate(abc.ABC):
    """A cell-like object that performs coordinate maps on it.
    """

    @abc.abstractmethod
    def jacobian_degree(self) -> int:
        """Degree of the Jacobian determinant.
        """

    @abc.abstractmethod
    def local_to_global(self, x_local: float) -> float:
        """Coordinate transform from local to global.
        """

    @abc.abstractmethod
    def global_to_local(self, x_global: float) -> float:
        """Coordinate transform from global to local.
        """

    @abc.abstractmethod
    def local_to_jacobian(self, x_local: float) -> float:
        """Get the Jacobian value at a point given by its local coordinate.
        """

    @abc.abstractmethod
    def global_to_jacobian(self, x_global: float) -> float:
        """Get the Jacobian value at a point given by its global coordinate.
        """

    def x_left(self):
        """Get the coordinate of this element's left boundary.
        """
        return self.local_to_global(-1)

    def x_right(self):
        """Get the coordinate of this element's right boundary.
        """
        return self.local_to_global(+1)

    def x_center(self):
        """Get the coordinate of this element's centroid.
        """
        return (self.x_right() + self.x_left()) / 2

    def length(self):
        """Get the length of this element.
        """
        return self.x_right() - self.x_left()


class Integrator(abc.ABC):
    """A cell-like object that performs integrations on it.
    """

    @abc.abstractstaticmethod
    def get_local_points(n_point) -> np.ndarray:
        """Get the local coordinates of a given number of quadrature points.
        """

    @abc.abstractstaticmethod
    def get_local_weights(n_point) -> np.ndarray:
        """Get the local weights of a given number of quadrature points.
        """

    def __init__(self, coordinate: Coordinate, n_point: int) -> None:
        assert n_point > 0
        self._coordinate = coordinate
        points = self.get_local_points(n_point)
        weights = self.get_local_weights(n_point)
        for i in range(n_point):
            local_i = points[i]
            points[i] = coordinate.local_to_global(local_i)
            weights[i] *= coordinate.local_to_jacobian(local_i)
        self._global_points = points
        self._global_weights = weights

    def coordinate(self) -> Coordinate:
        """Get the underlying coordinate mapping object.
        """
        return self._coordinate

    def n_point(self):
        return len(self._global_points)

    def get_global_points(self):
        """Get the global coordinates of all quadrature points.
        """
        return self._global_points

    def get_global_weights(self):
        """Get the global weights of all quadrature points.
        """
        return self._global_weights

    def integrate(self, function: callable, index=False):
        """Integrate a function defined in global coordinates or node index.
        """
        points = self.get_global_points()
        weights = self.get_global_weights()
        if index:
            value = weights[0] * function(0)
            for i in range(1, self.n_point()):
                value += weights[i] * function(i)
        else:
            value = weights[0] * function(points[0])
            for i in range(1, self.n_point()):
                value += weights[i] * function(points[i])
        return value

    def average(self, function: callable):
        """Get the average value of a function on this element.
        """
        integral = self.integrate(function)
        return integral / self.coordinate().length()

    def norm_1(self, function: callable):
        """Get the L_1 norm of a function on this element.
        """
        value = self.integrate(
            lambda x_global: np.abs(function(x_global)))
        return value

    def norm_2(self, function: callable):
        """Get the L_2 norm of a function on this element.
        """
        value = self.integrate(
            lambda x_global: np.abs(function(x_global))**2)
        return np.sqrt(value)

    def norm_infty(self, function: callable, n_point: int):
        """Get the (approximate) L_infty norm of a function on this element.
        """
        points = np.linspace(
            self.coordinate().x_left(), self.coordinate().x_right(), n_point)
        # Alternatively, quadrature points can be used.
        value = np.abs(function(points[0]))
        if np.isscalar(value):
            for i in range(1, len(points)):
                value = max(value, np.abs(function(points[i])))
        else:
            for i in range(1, len(points)):
                new_value = np.abs(function(points[i]))
                for j in range(len(value)):
                    value[j] = max(value[j], new_value[j])
        return value

    def inner_product(self, phi: callable, psi: callable):
        value = self.integrate(
            lambda x_global: phi(x_global) * psi(x_global))
        return value


class Expansion(abc.ABC):
    """The polynomial approximation of a general function.
    """

    def __init__(self, integrator: Integrator, value_type=float) -> None:
        self._integrator = integrator
        self._value_type = value_type

    def value_type(self):
        """Get the type of value.
        """
        return self._value_type

    @abc.abstractmethod
    def scalar_type(self):
        """Get the type of scalars.
        """

    def coordinate(self) -> Coordinate:
        """Get a refenece to the underlying Coordinate object.
        """
        return self._integrator.coordinate()

    def integrator(self) -> Integrator:
        """Get a refenece to the underlying Integrator object.
        """
        return self._integrator

    def x_left(self):
        return self.coordinate().x_left()

    def x_right(self):
        return self.coordinate().x_right()

    def x_center(self):
        return self.coordinate().x_center()

    def length(self):
        return self.coordinate().length()

    @abc.abstractmethod
    def name(self, verbose: bool) -> str:
        """Get the name of the expansion.
        """

    @abc.abstractmethod
    def n_term(self):
        """Number of terms in the approximated function.
        """

    @abc.abstractmethod
    def degree(self):
        """The highest degree of the approximated function.
        """

    @abc.abstractmethod
    def approximate(self, function: callable):
        """Approximate a general function.
        """

    @abc.abstractmethod
    def average(self):
        """Get the average value of the approximated function.
        """

    @abc.abstractmethod
    def get_basis(self, i_basis: int) -> callable:
        """Get i-th basis function, which maps x_global to its value.
        """

    @abc.abstractmethod
    def get_basis_values(self, x_global: float) -> np.ndarray:
        """Get the values of all basis functions at a given point.
        """

    @abc.abstractmethod
    def get_basis_gradients(self, x_global: float) -> np.ndarray:
        """Get the gradients of all basis functions at a given point.
        """

    @abc.abstractmethod
    def get_basis_hessians(self, x_global: float) -> np.ndarray:
        """Get the hessians of all basis functions at a given point.
        """

    @abc.abstractmethod
    def get_basis_derivatives(self, x_global: float, k: int) -> np.ndarray:
        """Get the derivatives of all basis functions at a given point.

        values[l] = the k-th derivative of l-th basis.
        """

    @abc.abstractmethod
    def get_basis_innerproducts(self):
        """Get the inner-products of each pair of basis functions.
        """

    @abc.abstractmethod
    def global_to_value(self, x_global: float):
        """Evaluate the approximation and return-by-value the result.
        """

    @abc.abstractmethod
    def global_to_gradient(self, x_global: float):
        """Evaluate the gradient of the approximation and return-by-value the result.
        """

    @abc.abstractmethod
    def global_to_derivatives(self, x_global: float, k: int):
        """Evaluate the kth derivatives of u^h and return-by-value the result.
        """

    @abc.abstractmethod
    def set_coeff(self, coeff: np.ndarray):
        """Set the coefficient for each basis.
        """

    @abc.abstractmethod
    def get_coeff_ref(self) -> np.ndarray:
        """Get the reference to the coefficient for each basis.
        """

    def get_boundary_basis_derivatives(self, k: int, left: bool):
        """Get the kth derivatives of basis at x_left or x_right.
        """
        if left:
            x = self.x_left()
        else:
            x = self.x_right()
        return self.get_basis_derivatives(x, k)

    def get_boundary_derivatives(self, k: int, left: bool, basis: bool):
        """Get the kth derivatives of basis or u^h at x_left or x_right.
        """
        if left:
            basis_derivatives = self.get_boundary_basis_derivatives(k, True)
        else:
            basis_derivatives = self.get_boundary_basis_derivatives(k, False)
        if basis:
            return basis_derivatives
        else:
            return self.get_coeff_ref().dot(basis_derivatives)


class Equation(abc.ABC):
    """A PDE in the form of \f$ \partial_t U + \partial_x F = \partial_x G + 0\f$.
    """

    def __init__(self, value_type) -> None:
        self._value_type = value_type

    def value_type(self):
        """Get the type of value.
        """
        return self._value_type

    @abc.abstractmethod
    def n_component(self):
        """Number of scalar components in U.
        """

    @abc.abstractmethod
    def component_names(self) -> tuple[str]:
        """Get the names of components in U.
        """

    @abc.abstractmethod
    def name(self, verbose: bool) -> str:
        """Get the name of the equation.
        """

    @abc.abstractmethod
    def get_convective_flux(self, u_given):
        """Get the value of F(U) for a given U.
        """

    @abc.abstractmethod
    def get_convective_radius(self, u_given):
        """Get the max convective speed.
        """

    def get_diffusive_coeff(self, u_given):
        """Get the value of b(U) for a given U if G(U, ∂U/∂x) = b(U) * ∂U/∂x.
        """
        return 0.0

    @abc.abstractmethod
    def get_diffusive_flux(self, u, du_dx, nu_extra):
        """Get the value of G(U, ∂U/∂x) for a given pair of U and ∂U/∂x, and optionally some extra viscosity.
        """

    @abc.abstractmethod
    def get_diffusive_radius(self, u_given, nu_extra, h_given):
        """Get the max equivalent speed of diffusion.
        """

    @abc.abstractmethod
    def inviscid(self):
        """Whether the equation has a viscous term.
        """


class ScalarEquation(Equation):

    def n_component(self):
        return 1

    def component_names(self) -> tuple[str]:
        return ('U',)

    @abc.abstractmethod
    def get_convective_speed(self, u_given):
        """Get the value of convective speed for a given U.
        """

    def get_convective_radius(self, u_given):
        return np.abs(self.get_convective_speed(u_given))

    def get_diffusive_flux(self, u, du_dx, nu_extra):
        return (self.get_diffusive_coeff(u) + nu_extra) * du_dx

    def get_diffusive_radius(self, u_given, nu_extra, h_given):
        nu = self.get_diffusive_coeff(u_given) + nu_extra
        return nu / h_given


class EquationSystem(Equation):

    def __init__(self) -> None:
        Equation.__init__(self, np.ndarray)

    @abc.abstractmethod
    def get_convective_jacobian(self, u_given) -> np.ndarray:
        """Get the value of ∂F(U)/∂U for a given U.
        """

    @abc.abstractmethod
    def get_convective_eigvals(self, u_given) -> np.ndarray:
        """Get the eigenvalues of ∂F(U)/∂U for a given U.
        """

    def get_convective_radius(self, u_given):
        eigvals = self.get_convective_eigvals(u_given)
        return np.max(np.abs(eigvals))

    @abc.abstractmethod
    def get_convective_eigmats(self, u_given) -> tuple[np.ndarray, np.ndarray]:
        """Get the left and right eigenmatrices of ∂F(U)/∂U for a given U.
        """

    def to_characteristics(self, u_given) -> np.ndarray:
        L, _ = self.get_convective_eigmats(u_given)
        return L @ u_given

    def from_characteristics(self, v_given) -> np.ndarray:
        _, R = self.get_convective_eigmats(v_given)
        return R @ v_given


class RiemannSolver(abc.ABC):
    """An exact or approximate solver for the Riemann problem of an Equation.
    """

    def __init__(self, equation: Equation) -> None:
        self._equation = equation

    def equation(self) -> Equation:
        return self._equation

    def value_type(self):
        return self.equation().value_type()

    @abc.abstractmethod
    def get_value(self, x_global, t_curr):
        """Get the solution of a Riemann problem.
        """

    @abc.abstractmethod
    def get_upwind_flux(self, u_left, u_right):
        """Get the value of the convective flux on the interface.
        """

    def get_upwind_value(self):
        """Get the value of the Riemann solution on the interface.

        The `get_upwind_flux()` method should be called just before calling this method.
        """
        return self.get_value(0, 1)

    # TODO: move to a DDG class
    @abc.abstractmethod
    def get_interface_gradient(self, h_left: float, h_right: float,
            u_jump, du_mean, ddu_jump):
        """Get the value of ∂U/∂x on the interface.
        """

    @abc.abstractmethod
    def diffusive_name(self) -> str:
        """Get the string representation of the diffusive solver.
        """

    @abc.abstractmethod
    def get_interface_flux_and_bjump(self, left, right):
        """Get the value of (F - G) and half-jump of b = nu*U on the interface.
        
        It is assumed G is in the form of nu * ∂U/∂x.
        """

    def get_interface_flux(self, left, right):
        """Get the value of (F - G) on the interface.
        
        It is assumed G is in the form of nu * ∂U/∂x.
        """
        flux, _ = self.get_interface_flux_and_bjump(left, right)
        return flux


class Face(abc.ABC):
    """An object for carrying a given state or the Riemann solution between two elements.
    """

    def __init__(self, riemann: RiemannSolver) -> None:
        self._riemann = riemann
        self._state = None
        self._flux = None
        self._jump = None

    @abc.abstractmethod
    def solve(self):
        """Solve the Riemann problem on this face.
        """

    def u(self):
        """Get the state on this face.
        """
        return self._state

    def flux(self):
        """Get the flux on this face.
        """
        return self._flux

    def jump(self):
        """Get the jump on this face.
        """
        return self._jump

    def get_convective_eigmats(self):
        """Get the eigen-matrices computed by the Riemann solution on this Face.
        """
        eq = self._riemann.equation()
        return EquationSystem.get_convective_eigmats(eq, self.u())


class Element(abc.ABC):
    """A cell-like object that carries some expansion of the solution.
    """
    # TODO: rename to PhysicalElement

    def __init__(self, riemann: RiemannSolver, expansion: Expansion) -> None:
        assert isinstance(riemann, RiemannSolver)
        self._riemann = riemann
        self._expansion = expansion
        self._coordinate = expansion.coordinate()
        self._integrator = expansion.integrator()
        self._left_face = None
        self._right_face = None
        self._left_expansion = None
        self._right_expansion = None
        self._extra_viscosity = None
        if isinstance(self.equation(), EquationSystem):
            self._eigen_matrices = None

    def riemann(self) -> RiemannSolver:
        """Get a refenece to the underlying RiemannSolver object.
        """
        return self._riemann

    def equation(self) -> Equation:
        """Get a refenece to the underlying Equation object.
        """
        return self._riemann.equation()

    def is_scalar(self):
        return isinstance(self.equation(), ScalarEquation)

    def is_system(self):
        return isinstance(self.equation(), EquationSystem)

    def value_type(self):
        return self.expansion().value_type()

    def scalar_type(self):
        return self.expansion().scalar_type()

    def coordinate(self) -> Coordinate:
        """Get a refenece to the underlying Coordinate object.
        """
        return self._coordinate

    def integrator(self) -> Integrator:
        """Get a refenece to the underlying Integrator object.
        """
        return self._integrator

    def expansion(self) -> Expansion:
        """Get a refenece to the underlying Expansion object.
        """
        return self._expansion

    def get_faces(self) -> tuple[Face, Face]:
        return self._left_face, self._right_face

    def neighbor_expansions(self) -> tuple[Expansion, Expansion]:
        return self._left_expansion, self._right_expansion

    def x_left(self):
        return self.coordinate().x_left()

    def x_right(self):
        return self.coordinate().x_right()

    def x_center(self):
        return self.coordinate().x_center()

    def length(self):
        return self.coordinate().length()

    def degree(self):
        """The highest degree of the approximated solution.
        """
        return self.expansion().degree()

    def n_term(self):
        """Number of terms in the approximated function.
        """
        return self.expansion().n_term()

    def n_dof(self):
        """Count degrees of freedom in current object.
        """
        return self.n_term() * self.equation().n_component()

    def approximate(self, function: callable):
        """Approximate a general function as u^h.
        """
        self.expansion().approximate(function)
        if self.is_system():
            self._update_eigen_matrices()

    def _build_mass_matrix(self):
        mass_matrix = self.expansion().get_basis_innerproducts()
        return mass_matrix

    def get_extra_viscosity(self, x_global: float) -> float:
        if callable(self._extra_viscosity):
            return self._extra_viscosity(x_global)
        elif isinstance(self._extra_viscosity, float):
            return self._extra_viscosity
        else:
            assert not self._extra_viscosity
            if self.equation().n_component() == 1:
                return 0.0
            else:
                return np.zeros(self.equation().n_component())

    def get_dg_flux(self, x_global, u_given=None, du_given=None):
        """Get the value of f(u^h, du^h) at a given point.
        """
        if u_given is None:
            u_approx = self.get_solution_value(x_global)
        else:
            u_approx = u_given
        flux = self.equation().get_convective_flux(u_approx)
        if du_given is None:
            du_approx = self.expansion().global_to_gradient(x_global)
        else:
            du_approx = du_given
        if self._extra_viscosity or not self.equation().inviscid():
            flux -= self.equation().get_diffusive_flux(u_approx, du_approx,
                self.get_extra_viscosity(x_global))
        return flux

    def get_solution_value(self, x_global: float):
        """Get the value of u^h at a given point.
        """
        return self.expansion().global_to_value(x_global)

    def get_solution_gradient(self, x_global: float):
        """Get the gradient value of u^h at a given point.
        """
        return self.expansion().global_to_gradient(x_global)

    def set_solution_coeff(self, column):
        """Set coefficients of the solution's expansion.

        The element is responsible for the column-to-coeff conversion.
        """
        if not self.is_system():
            self.expansion().set_coeff(column)
        else:
            n_row = self.n_term()
            n_col = self.equation().n_component()
            coeff = np.ndarray(n_row, np.ndarray)
            for i_row in range(n_row):
                coeff[i_row] = np.ndarray(n_col, self.scalar_type())
            i_dof = 0
            for i_col in range(n_col):
                for i_row in range(n_row):
                    coeff[i_row][i_col] = column[i_dof]
                    i_dof += 1
            assert i_dof == len(column)
            self.expansion().set_coeff(coeff)
            self._update_eigen_matrices()

    def get_solution_column(self) -> np.ndarray:
        """Get coefficients of the solution's expansion.

        The element is responsible for the coeff-to-column conversion.
        """
        coeff = self.expansion().get_coeff_ref()
        if coeff.dtype == np.ndarray:
            column = np.ndarray(self.n_dof(), self.scalar_type())
            n_row = len(coeff)
            n_col = len(coeff[0])
            i_dof = 0
            for i_col in range(n_col):
                for i_row in range(n_row):
                    column[i_dof] = coeff[i_row][i_col]
                    i_dof += 1
            assert i_dof == len(column)
            return column
        else:
            return coeff

    @abc.abstractmethod
    def divide_mass_matrix(self, column: np.ndarray):
        """Divide the mass matrix of this element by the given column.

        Solve the system of linear equations Ax = b, in which A is the mass matrix of this element.
        """

    @abc.abstractmethod
    def get_interior_residual(self) -> np.ndarray:
        """Get the residual given by the flux in the element.
        """

    @abc.abstractmethod
    def add_interface_residual(self, left_flux, right_flux,
            residual: np.ndarray):
        """Add the residual given by the flux on the interface.
        """

    def get_interface_correction(self, left_jump, right_jump) -> np.ndarray:
        """Get the interface correction for the DDG method.

        See [Liu and Yan, "The Direct Discontinuous Galerkin (DDG) Method for Diffusion with Interface Corrections", Communications in Computational Physics 8, 3 (2010), pp. 541--564](https://doi.org/10.4208/cicp.010909.011209a) for details.

        The jumps passed in have already been divided by 2.
        """
        e = self.expansion()
        correction = np.tensordot(
            e.get_boundary_derivatives(1, True, True), left_jump, 0)
        correction += np.tensordot(
            e.get_boundary_derivatives(1, False, True), right_jump, 0)
        return correction

    def add_interface_correction(self, left_jump, right_jump, residual: np.ndarray):
        """Add the interface correction to the residual.
        """
        pass

    @abc.abstractmethod
    def suggest_cfl(self, rk_order: int) -> float:
        """Suggest a CFL number for explicit RK time stepping.
        """

    def _suggest_dt_by_blazek(self):
        """See Eq. (6.20) in Blazek (2015).
        """
        h = self.length()
        p = self.degree()
        spatial_factor = 3 * p**2  # may be too large
        # The following values are tested for nu <= 1.0 and t <= 1.0.
        if p == 0:
            spatial_factor = 1
        elif p == 1:
            spatial_factor = 4
        elif p == 2:
            spatial_factor = 6
        elif p == 3:
            spatial_factor = 12
        elif p == 4:
            spatial_factor = 28
        elif p == 5:
            spatial_factor = 50
        elif p == 6:
            spatial_factor = 90
        if True:
            u = self.expansion().average()
            lambda_c = self.equation().get_convective_radius(u)
            lambda_c = max(lambda_c, 1e-16)
            nu = self.get_extra_viscosity(self.x_center())
            lambda_d = self.equation().get_diffusive_radius(u, nu, h)
            delta_t = h / (lambda_c + spatial_factor * lambda_d)
            return delta_t * self.suggest_cfl(3)
        points = np.linspace(self.x_left(), self.x_right(), self.n_term()+1)
        delta_t = np.infty
        for x in points:
            u = self.get_solution_value(x)
            lambda_c = self.equation().get_convective_radius(u)
            lambda_c = max(lambda_c, 1e-16)
            nu = self.get_extra_viscosity(x)
            lambda_d = self.equation().get_diffusive_radius(u, nu, h)
            delta_t = min(delta_t, h / (lambda_c + spatial_factor * lambda_d))
        return delta_t * self.suggest_cfl(3)

    def _suggest_dt_by_klockner(self):
        """See Eq. (2.1) in Klöckner (2011).
        """
        h = self.length()
        p2 = self.degree()**2
        if True:
            u = self.expansion().average()
            lambda_c = self.equation().get_convective_radius(u)
            lambda_c = max(lambda_c, 1e-16)
            nu = self.get_extra_viscosity(self.x_center())
            lambda_d = self.equation().get_diffusive_radius(u, nu, h)
            delta_t = h / p2 / (lambda_c + p2 * lambda_d)
            return delta_t
        points = np.linspace(self.x_left(), self.x_right(), self.n_term()+1)
        delta_t = np.infty
        for x in points:
            u = self.get_solution_value(x)
            lambda_c = self.equation().get_convective_radius(u)
            lambda_c = max(lambda_c, 1e-16)
            nu = self.get_extra_viscosity(x)
            lambda_d = self.equation().get_diffusive_radius(u, nu, h)
            delta_t = min(delta_t, h / p2 / (lambda_c + p2 * lambda_d))
        return delta_t

    def suggest_delta_t(self):
        return self._suggest_dt_by_blazek()

    def _update_eigen_matrices(self):
        u_average = self.expansion().average()
        equation = self.equation()
        assert isinstance(equation, EquationSystem)
        self._eigen_matrices = equation.get_convective_eigmats(u_average)

    def get_convective_eigmats(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the eigen-matrices computed by the average value on this Element.
        """
        return self._eigen_matrices


class Interface(Face):
    """The common interface of two elements.
    """

    def __init__(self, left: Element, right: Element):
        Face.__init__(self, left.riemann())
        self._left = left
        self._right = right

    def solve(self):
        self._flux, self._jump = \
            self._riemann.get_interface_flux_and_bjump(self._left, self._right)
        self._state = self._riemann.get_upwind_value()


class Inlet(Face):
    """A Face object at the boundary with a given state.
    """

    def __init__(self, u_given, element: Element, left: bool):
        Face.__init__(self, element.riemann())
        self._u_given = u_given
        self._expansion = element.expansion()
        self._left = left
        self._jump = 0 * u_given

    def solve(self):
        if self._left:
            u_left = self._u_given
            u_right = self._expansion.get_boundary_derivatives(0, True, False)
        else:
            u_right = self._u_given
            u_left = self._expansion.get_boundary_derivatives(0, False, False)
        self._flux = self._riemann.get_upwind_flux(u_left, u_right)
        self._state = self._riemann.get_upwind_value()


class Grid(abc.ABC):

    def __init__(self, n_element: int):
        self._elements = np.ndarray(n_element, Element)
        self._faces = np.ndarray(n_element + 1, Face)
        self._periodic = True

    @abc.abstractmethod
    def get_element_index(self, point) -> int:
        """Get the index of the element in which the given point locates.
        """

    def get_face_by_index(self, i: int) -> Face:
        """Get the `Face` object at the left of the `i`th `Element`.
        """
        return self._faces[i]

    def get_element_by_index(self, i: int) -> Element:
        """Get the `i`th `Element` object.
        """
        return self._elements[i]

    def get_element(self, point) -> Element:
        """Get the element in which the given point locates.
        """
        return self._elements[self.get_element_index(point)]

    def x_left(self):
        """Get the coordinate of this object's left boundary.
        """
        return self.get_element_by_index(0).x_left()

    def x_right(self):
        """Get the coordinate of this object's right boundary.
        """
        return self.get_element_by_index(-1).x_right()

    def length(self):
        """Get the length of this object.
        """
        return self.x_right() - self.x_left()

    def n_element(self):
        """Count elements in this object.
        """
        return len(self._elements)

    def delta_x(self, i_cell: int):
        """Get the length of the i-th element.
        """
        return self.get_element_by_index(i_cell).length()

    def value_type(self):
        return self.get_element_by_index(0).value_type()

    def build_interface(self, i: int):
        """Set the `Interface` at the left of `i`th `Element`.
        """
        right = self.get_element_by_index(i)
        left = self.get_element_by_index(i - 1)
        face = Interface(left, right)
        right._left_face = face
        left._right_face = face
        self._faces[i] = face
        if i == 0:
            self._faces[-1] = face

    def set_boundary_values(self, u_left, u_right):
        """Set prescribed (far-field) boundary values.
        """
        self._periodic = False
        assert u_left is None or isinstance(u_left, self.value_type())
        assert u_right is None or isinstance(u_right, self.value_type())
        right = self.get_element_by_index(0)
        face = Inlet(u_left, right, True)
        right._left_face = face
        self._faces[0] = face
        left = self.get_element_by_index(-1)
        face = Inlet(u_right, left, False)
        left._right_face = face
        self._faces[-1] = face
        self.link_neighbors()

    def is_periodic(self):
        """Whether a periodic boundary condition is applied.
        """
        return self._periodic

    @abc.abstractmethod
    def link_neighbors(self):
        """Link each element to its neighbors' expansions.
        """


class Detector(abc.ABC):
    """An object that detects jumps on an element.
    """

    @abc.abstractmethod
    def name(self, verbose: bool) -> str:
        """Get the name of the detector.
        """

    @abc.abstractmethod
    def get_troubled_cell_indices(self, grid: Grid):
        """Whether the current element is troubled.
        """


class Limiter(abc.ABC):
    """An object that limits oscillations on an element.
    """

    @abc.abstractmethod
    def name(self, verbose: bool) -> str:
        """Get the name of the limiter.
        """

    @abc.abstractmethod
    def reconstruct(self, troubled_cell_indices, grid: Grid):
        """Reconstruct the expansion on each troubled cell.
        """


class Viscosity(abc.ABC):
    """An object that adds artificial viscosity to a spatial scheme.
    """

    def __init__(self) -> None:
        self._index_to_coeff = dict()

    @abc.abstractmethod
    def name(self, verbose: bool) -> str:
        """Get the name of the limiter.
        """

    @abc.abstractmethod
    def generate(self, troubled_cell_indices, grid: Grid):
        """Generate artificial viscosity for each troubled cell.
        """

    def get_cell_viscosity(self, i_cell: int):
        """Get the viscosity of the ith cell.
        """
        assert 0 <= i_cell
        if i_cell in self._index_to_coeff:
            return self._index_to_coeff[i_cell]
        else:
            return None


class OdeSystem(abc.ABC):
    """A system of ODEs in the form of dU/∂t = R.
    """

    @abc.abstractmethod
    def set_solution_column(self, column):
        """Overwrite the values in the solution column.
        """

    @abc.abstractmethod
    def get_solution_column(self):
        """Get a copy of the solution column.
        """

    @abc.abstractmethod
    def get_residual_column(self):
        """Get a copy of the residual column.
        """

    def set_time(self, t_curr: float):
        """Set the current time value.

        For steady system, it does nothing.
        """


class OdeSolver(abc.ABC):
    """A solver for the standard ODE system dU/dt = R.

    The solution column U and the residual column R is provided by a OdeSystem object.
    """

    @abc.abstractmethod
    def update(self, ode_system: OdeSystem, delta_t: float, t_curr: float):
        """Update the given OdeSystem object to the next time step.
        """


class SpatialScheme(Grid, OdeSystem):
    """An ODE system given by some FiniteElement scheme.
    """

    def __init__(self, riemann: RiemannSolver,
            n_element: int, x_left: float, x_right: float,
            detector=None, limiter=None, viscosity=None) -> None:
        assert x_left < x_right
        assert n_element > 1
        Grid.__init__(self, n_element)
        self._riemann = riemann
        self._detector = detector
        self._limiter = limiter
        self._viscosity = viscosity

    def value_type(self):
        return self.get_element_by_index(0).value_type()

    def scalar_type(self):
        return self.get_element_by_index(0).scalar_type()

    def degree(self):
        """Get the degree of the polynomial approximation.
        """
        return self.get_element_by_index(0).degree()

    def equation(self) -> Equation:
        return self._riemann.equation()

    @abc.abstractmethod
    def name(self, verbose: bool) -> str:
        """Get the compact string representation of the method.
        """

    @abc.abstractmethod
    def n_dof(self):
        """Count degrees of freedom in current object.
        """

    @abc.abstractmethod
    def initialize(self, function: callable,
            temp_detector = None, temp_limiter = None):
        """Set the initial condition.
        """

    @abc.abstractmethod
    def get_solution_value(self, point):
        """Get the solution value at a given point.
        """

    @abc.abstractmethod
    def get_flux_value(self, point):
        """Get the flux value at a given point.
        """

    def viscosity(self) -> Viscosity:
        """Get a reference to the underlying viscosity model.
        """
        return self._viscosity

    def set_detector_and_limiter(self, detector, limiter, viscosity):
        if isinstance(detector, Detector):
            self._detector = detector
        if isinstance(limiter, Limiter):
            self._limiter = limiter
        if isinstance(viscosity, Viscosity):
            self._viscosity = viscosity

    def apply_limiter(self):
        if self._detector and self._limiter:
            indices = self._detector.get_troubled_cell_indices(self)
            self._limiter.reconstruct(indices, self)

    def apply_viscosity(self):
        if self._detector and self._viscosity:
            indices = self._detector.get_troubled_cell_indices(self)
            self._viscosity.generate(indices, self)
            for i_cell in range(self.n_element()):
                self.get_element_by_index(i_cell)._extra_viscosity = \
                    self._viscosity.get_cell_viscosity(i_cell)

    def suggest_delta_t(self, delta_t):
        for i_cell in range(self.n_element()):
            cell_i = self.get_element_by_index(i_cell)
            delta_t = min(delta_t, cell_i.suggest_delta_t())
        return delta_t


if __name__ == '__main__':
    pass
