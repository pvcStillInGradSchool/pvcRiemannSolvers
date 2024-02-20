"""Implement some special polynomials.
"""
import numpy as np
from scipy.special import eval_legendre, roots_legendre

from concept import Polynomial


class Radau(Polynomial):
    r"""The left and right Radau polynomials.

    The \f$ k \f$-degree left and right Radau polynomial are defined as
      \f[
        \mathrm{R}_{k,\mathrm{left}}(\xi) = \frac{\mathrm{P}_{k}(\xi)}{2} 
                                 + \frac{\mathrm{P}_{k-1}(\xi)}{2},\quad
        \mathrm{R}_{k,\mathrm{right}}(\xi) = \mathrm{R}_{k,\mathrm{left}}(-\xi),
      \f]
    where \f$ \mathrm{P}_{k} \f$ is the \f$ k \f$-degree Legendre polynomial.
    """

    def __init__(self, degree: int) -> None:
        super().__init__()
        assert 1 <= degree
        self._k = degree

    def name(self, verbose=True) -> str:
        return f'Radau({self.p_str()})'

    def degree(self):
        return self._k

    def local_to_value(self, x_local):
        legendre_value_curr = eval_legendre(self._k, x_local)
        legendre_value_prev = eval_legendre(self._k-1, x_local)
        left = (legendre_value_curr + legendre_value_prev) / 2
        right = legendre_value_curr - legendre_value_prev
        right *= (-1)**self._k / 2
        return (left, right)

    def local_to_gradient(self, x_local):
        legendre_derivative_prev = 0.0
        legendre_derivative_curr = 0.0
        for k in range(1, self._k + 1):
            legendre_derivative_prev = legendre_derivative_curr
            # prev == k - 1, curr == k
            legendre_derivative_curr = k * eval_legendre(k-1, x_local)
            legendre_derivative_curr += x_local * legendre_derivative_prev
        left = (legendre_derivative_curr + legendre_derivative_prev) / 2
        right = legendre_derivative_curr - legendre_derivative_prev
        right *= (-1)**self._k / 2
        return (left, right)


class Huynh(Polynomial):
    r"""The left and right \f$ g(\xi) \f$ in Huynh's FR schemes.

    The right \f$ g(\xi) \f$ with lumping conditions up to the \f$ m \f$th derivative for a \f$ k \f$-degree \f$ u^h \f$ is a \f$ (k+1) \f$-degree polynomial, which satisfies 
    (a) \f$ (m+1) \f$ lumping conditions \f$ g^{(0)}(-1) = \dots = g^{(m)}(-1) = 0 \f$, and 
    (b) \f$ (k-m) \f$ orthogonality conditions \f$ g \perp \mathbb{P}_{0} \land \cdots \land g \perp \mathbb{P}_{k-m-1} \f$, and
    (c) \f$ 1 \f$ boundary value condition \f$ g(1) = 1 \f$.
    """

    def __init__(self, degree: int, n_lump: int) -> None:
        super().__init__()
        assert 1 <= degree <= 10
        assert 1 <= n_lump <= degree
        self._n_lump = n_lump
        n_term = degree + 1
        a = np.zeros((n_term, n_term))
        b = np.zeros(n_term)
        # lumping conditions
        factorials = np.ones(n_term)
        for i in range(2, n_term):
            factorials[i] = factorials[i - 1] * i
        for l in range(n_lump):
            for i in range(n_term):
                if i >= l:
                    a[l][i] = (-1)**(i - l) * factorials[i] / factorials[i - l]
        # orthogonality conditions
        x, w = roots_legendre(degree)
        for l in range(n_lump, degree):
            p_legendre = l - n_lump
            for i in range(n_term):
                for q in range(len(x)):
                    a[l][i] += x[q]**i * eval_legendre(p_legendre, x[q]) * w[q]
        # boundary value condition
        a[degree] = 1
        b[degree] = 1
        self._value_coeff = np.linalg.solve(a, b)
        self._grad_coeff = b
        for i in range(n_term):
            self._grad_coeff[i] = self._value_coeff[i] * i
        self._value_powers = np.arange(n_term)
        self._grad_powers = np.arange(n_term) - 1
        self._grad_powers[0] = 0

    def name(self, verbose=True) -> str:
        return f'Huynh({self.p_str()}, lump={self._n_lump})'

    def degree(self):
        return len(self._value_coeff) - 1

    def _local_to_right_value(self, x_local):
        return self._value_coeff.dot(x_local ** self._value_powers)

    def local_to_value(self, x_local):
        left = self._local_to_right_value(-x_local)
        right = self._local_to_right_value(x_local)
        return (left, right)

    def _local_to_right_grad(self, x_local):
        return self._grad_coeff.dot(x_local ** self._grad_powers)

    def local_to_gradient(self, x_local):
        left = -self._local_to_right_grad(-x_local)
        right = self._local_to_right_grad(x_local)
        return (left, right)


class Vincent(Polynomial):
    r"""The left and right \f$ g(\xi) \f$ in Vincent's ESFR schemes.

    The right \f$ g(\xi) \f$ for a \f$ k \f$-degree \f$ u^h \f$ is defined as
      \f[
        g(\xi) = \frac{\mathrm{P}_{k}(\xi)}{2}
        + (1 - \mu) \frac{\mathrm{P}_{k-1}(\xi)}{2}
        +      \mu  \frac{\mathrm{P}_{k+1}(\xi)}{2},
      \f]
    where \f$ \mathrm{P}_{k} \f$ is the \f$ k \f$-degree Legendre polynomial.
    """

    @staticmethod
    def discontinuous_galerkin(k: int):
        return 1.0

    @staticmethod
    def spectral_difference(k: int):
        return (k + 1) / (2*k + 1)

    @staticmethod
    def huynh_lumping_lobatto(k: int):
        return k / (2*k + 1)

    def __init__(self, degree: int, mu: callable = huynh_lumping_lobatto) -> None:
        super().__init__()
        assert 1 <= degree <= 10
        self._k = degree - 1  # degree of solution
        self._next_ratio = mu(self._k)
        self._prev_ratio = 1 - self._next_ratio
        self._local_to_gradient = dict()

    def name(self, verbose=True) -> str:
        c = r'$c$'
        return f'Vincent({self.p_str()}, {c}={self._next_ratio:.2f})'

    def degree(self):
        return self._k + 1

    def local_to_value(self, x_local):
        def right(xi):
            val = eval_legendre(self._k, xi)
            val += self._prev_ratio * eval_legendre(self._k - 1, xi)
            val += self._next_ratio * eval_legendre(self._k + 1, xi)
            return val / 2
        return (right(-x_local), right(x_local))

    def local_to_gradient(self, x_local):
        if x_local in self._local_to_gradient:
            return self._local_to_gradient[x_local]
        legendre_derivative_prev = 0.0
        legendre_derivative_curr = 0.0
        legendre_derivative_next = 0.0
        for k_curr in range(1 + self._k):
            if k_curr > 0:
                legendre_derivative_prev = legendre_derivative_curr
                legendre_derivative_curr = legendre_derivative_next
            k_next = k_curr + 1
            legendre_derivative_next = (k_next * eval_legendre(k_curr, x_local)
                + x_local * legendre_derivative_curr)
        left = (-1)**self._k * (legendre_derivative_curr
            - self._prev_ratio * legendre_derivative_prev
            - self._next_ratio * legendre_derivative_next) / 2
        right = (legendre_derivative_curr
            + self._prev_ratio * legendre_derivative_prev
            + self._next_ratio * legendre_derivative_next) / 2
        self._local_to_gradient[x_local] = (left, right)
        return (left, right)


class IthLagrange(Polynomial):
    """The Ith Lagrange polynomial for N given points.
    """

    def __init__(self, index: int, points: np.ndarray) -> None:
        self._i = index
        assert 0 <= index < len(points)
        self._points = points

    def name(self, verbose=True) -> str:
        return f'Lagrange({self.p_str()}, i={self._i})'

    def n_point(self):
        return len(self._points)

    def degree(self):
        return len(self._points) - 1

    def local_to_value(self, x_local: float):
        value = 1.0
        for j in range(self.n_point()):
            if j == self._i:
                continue
            dividend = x_local - self._points[j]
            divisor = self._points[self._i] - self._points[j]
            value *= (dividend / divisor)
        return value

    def local_to_gradient(self, x_local: float):
        value = 0.0
        for j in range(self.n_point()):
            if j == self._i:
                continue
            dividend = 1.0
            divisor = self._points[self._i] - self._points[j]
            for k in range(self.n_point()):
                if k in (self._i, j):
                    continue
                dividend *= x_local - self._points[k]
                divisor *= self._points[self._i] - self._points[k]
            value += dividend / divisor
        return value


class LagrangeBasis(Polynomial):
    """All Lagrange polynomials, which form a basis.
    """

    def __init__(self, points: np.ndarray) -> None:
        super().__init__()
        n_point = len(points)
        self._points = points
        self._lagranges = np.ndarray(n_point, IthLagrange)
        for i in range(n_point):
            self._lagranges[i] = IthLagrange(i, self._points)

    def name(self, verbose=True) -> str:
        return f'LagrangeBasis({self.p_str()})'

    def n_term(self):
        return len(self._points)

    def degree(self):
        return len(self._points) - 1

    def local_to_value(self, x_local):
        values = np.ndarray(self.n_term())
        for i in range(self.n_term()):
            values[i] = self._lagranges[i].local_to_value(x_local)
        return values

    def local_to_gradient(self, x_local):
        values = np.ndarray(self.n_term())
        for i in range(self.n_term()):
            values[i] = self._lagranges[i].local_to_gradient(x_local)
        return values


if __name__ == '__main__':
    pass
