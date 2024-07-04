"""Tests for some temporal schemes.
"""
import unittest
import numpy as np
from matplotlib_wrapper import savefig
from matplotlib import pyplot as plt

from concept import OdeSystem
from temporal import ExplicitEuler, RungeKutta


class Constant(OdeSystem):
    """The simplest implementation of OdeSystem.
    """
    def __init__(self, a_mat) -> None:
        n_row, n_col = a_mat.shape
        assert n_row == n_col
        self._n = n_col
        self._a = a_mat.copy()
        self._u = np.zeros(n_col)

    def set_solution_column(self, column):
        for i in range(self._n):
            self._u[i] = column[i]

    def get_solution_column(self):
        return self._u.copy()

    def get_residual_column(self):
        residual = self._a.dot(self._u)
        return residual


class Pendulum(OdeSystem):
    """A simple ODE system obtained from u'' + u = 0.
    """

    def __init__(self, u_init, v_init):
        self._u_init = u_init
        self._v_init = v_init
        self._solution = np.array((u_init, v_init))

    def reset(self):
        self._solution = np.array((self._u_init, self._v_init))

    def get_exact_solution(self, time):
        u = self._u_init * np.cos(time)
        v = self._v_init * np.sin(time)
        return u + v

    def set_solution_column(self, column):
        self._solution[0] = column[0]
        self._solution[1] = column[1]

    def get_solution_column(self):
        return self._solution.copy()

    def get_residual_column(self):
        u = self._solution[0]
        v = self._solution[1]
        return np.array((v, -u))


class TestExplicitEuler(unittest.TestCase):
    """Test the ExplicitEuler class.
    """

    def test_update(self):
        """Test the update() method.
        """
        n_row = 10
        a_const = np.random.rand(n_row, n_row)
        ode_system = Constant(a_const)
        u_curr = np.random.rand(n_row)
        ode_system.set_solution_column(u_curr)
        scheme = ExplicitEuler()
        delta_t = 0.01
        scheme.update(ode_system, delta_t, t_curr=0.0)
        u_expect = u_curr + a_const.dot(u_curr) * delta_t
        u_actual = ode_system.get_solution_column()
        self.assertFalse(np.linalg.norm(u_expect - u_actual))


class TestRungeKutta(unittest.TestCase):
    """Test the RungeKutta class.
    """

    def euler(self, solution, a_const, delta_t):
        """Wrap the computation of the explicit Euler method.
        """
        return solution + a_const.dot(solution) * delta_t

    def test_rk1_update(self):
        """Test RK1's update() method.
        """
        n_row = 10
        a_const = np.random.rand(n_row, n_row)
        ode_system = Constant(a_const)
        u_curr = np.random.rand(n_row)
        ode_system.set_solution_column(u_curr)
        scheme = RungeKutta(order=1)
        delta_t = 0.01
        scheme.update(ode_system, delta_t, t_curr=0.0)
        u_actual = ode_system.get_solution_column()
        u_expect = self.euler(u_curr, a_const, delta_t)
        self.assertFalse(np.linalg.norm(u_expect - u_actual))

    def test_rk2_update(self):
        """Test RK2's update() method.
        """
        n_row = 10
        a_const = np.random.rand(n_row, n_row)
        ode_system = Constant(a_const)
        u_curr = np.random.rand(n_row)
        ode_system.set_solution_column(u_curr)
        scheme = RungeKutta(order=2)
        delta_t = 0.01
        scheme.update(ode_system, delta_t, t_curr=0.0)
        u_actual = ode_system.get_solution_column()
        u_frac12 = self.euler(u_curr, a_const, delta_t)
        u_expect = (u_curr + self.euler(u_frac12, a_const, delta_t)) / 2
        self.assertFalse(np.linalg.norm(u_expect - u_actual))

    def test_rk3_update(self):
        """Test RK3's update() method.
        """
        n_row = 10
        a_const = np.random.rand(n_row, n_row)
        ode_system = Constant(a_const)
        u_curr = np.random.rand(n_row)
        ode_system.set_solution_column(u_curr)
        scheme = RungeKutta(order=3)
        delta_t = 0.01
        scheme.update(ode_system, delta_t, t_curr=0.0)
        u_actual = ode_system.get_solution_column()
        u_frac13 = self.euler(u_curr, a_const, delta_t)
        u_frac23 = (u_curr * 3 + self.euler(u_frac13, a_const, delta_t)) / 4
        u_expect = (u_curr + self.euler(u_frac23, a_const, delta_t) * 2) / 3
        self.assertFalse(np.linalg.norm(u_expect - u_actual))

    def test_plot(self):
        ode_system = Pendulum(u_init=-1.0, v_init=1.0)
        t_start = 0.0
        t_stop = 10 * np.pi
        n_step = 1001
        time_values = np.linspace(t_start, t_stop, n_step)
        delta_t = time_values[1] - time_values[0]
        exact_values = ode_system.get_exact_solution(time_values)
        markers = ('.', '1', '2', '3', '4')
        for order in (1, 2, 3, 4):
            ode_system.reset()
            scheme = RungeKutta(order)
            rk_values = np.ndarray(n_step)
            rk_values[0] = ode_system.get_solution_column()[0]
            for i_step in range(1, n_step):
                t_curr = t_start + delta_t * i_step
                scheme.update(ode_system, delta_t, t_curr)
                rk_values[i_step] = ode_system.get_solution_column()[0]
            plt.plot(time_values, rk_values, markers[order], label=f'RK{order}')
        plt.plot(time_values, exact_values, label='Exact')
        plt.legend()
        # plt.show()
        savefig('compare_runge_kutta')


if __name__ == '__main__':
    unittest.main()
