import argparse
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as mpla

import concept
import equation
import riemann
import spatial
import temporal
from matplotlib_wrapper import line_styles


class LinearAdvection(object):

    def __init__(self, a_const: float, k_low: float, k_high: float,
            degree: int, n_element: int, x_left: float, x_right: float,
            ode_solver: concept.OdeSolver) -> None:
        self._a_const = a_const
        self._degree = degree
        self._n_element = n_element
        self._x_left = x_left
        self._x_right = x_right
        riemann_object = riemann.LinearAdvection(a_const)
        self._dg = spatial.LegendreDG(riemann_object,
            degree, n_element, x_left, x_right)
        self._fr = spatial.LagrangeFR(riemann_object,
            degree, n_element, x_left, x_right)
        self._length = self._dg.length()
        self._delta_x = self._dg.delta_x(0)
        self._kappa_low = k_low * np.pi / self._delta_x
        self._kappa_high = k_high * np.pi / self._delta_x
        self._ode_solver = ode_solver

    def a_max(self):
        return np.abs(self._a_const)

    def get_exact_ydata(self, t_curr, points):
        n_point = len(points)
        expect_solution = np.ndarray(n_point)
        for i in range(n_point):
            point_i = points[i]
            expect_solution[i] = self.u_exact(point_i, t_curr)
        return expect_solution

    def get_approx_ydata(self, t_curr, points):
        n_point = len(points)
        dg_solution = np.ndarray(n_point)
        fr_solution = np.ndarray(n_point)
        for i in range(n_point):
            point_i = points[i]
            dg_solution[i] = self._dg.get_solution_value(point_i)
            fr_solution[i] = self._fr.get_solution_value(point_i)
        return dg_solution, fr_solution

    def animate(self, t_start: float, t_stop: float,  n_step: int):
        delta_x = self._delta_x
        delta_t = (t_stop - t_start) / n_step
        cfl = self.a_max() * delta_t / delta_x
        print(f"delta_x = {delta_x}, delta_t = {delta_t}, cfl = {cfl}")
        # general plot setting
        plt.figure(figsize=(6, 4))
        plt.ylim([-1.4, 1.4])
        plt.xticks(np.linspace(self._x_left, self._x_right,
            self._n_element + 1), minor=True)
        plt.xticks(np.linspace(self._x_left, self._x_right,
            3), minor=False)
        plt.grid(which='both')
        # initialize line-plot objects
        exact_points = np.linspace(self._x_left, self._x_right, 1001)
        approx_points = np.linspace(self._x_left, self._x_right, 101)
        expect_line, = plt.plot([], [], 'r-', label='Exact')
        dg_line, = plt.plot([], [], linestyle=line_styles[0][1],
            label=f'LegendreDG{self._degree+1}')
        fr_line, = plt.plot([], [], linestyle=line_styles[1][1],
            label=f'LagrangeFR{self._degree+1}')
        # initialize animation
        def init_func():
            u_init = lambda x_global: self.u_init(x_global)
            self._dg.initialize(u_init)
            self._fr.initialize(u_init)
            expect_line.set_xdata(exact_points)
            dg_line.set_xdata(approx_points)
            fr_line.set_xdata(approx_points)
        # update data for the next frame
        def func(t_curr):
            expect_solution = self.get_exact_ydata(t_curr, exact_points)
            expect_line.set_ydata(expect_solution)
            dg_solution, fr_solution = self.get_approx_ydata(t_curr, approx_points)
            dg_line.set_ydata(dg_solution)
            fr_line.set_ydata(fr_solution)
            plt.title(f't = {t_curr:.2f}')
            plt.legend(loc='upper right')
            self._ode_solver.update(self._dg, delta_t, t_curr)
            self._ode_solver.update(self._fr, delta_t, t_curr)
        frames = np.linspace(t_start, t_stop, n_step)
        anim = mpla.FuncAnimation(
            plt.gcf(), func, frames, init_func, interval=5)
        plt.show()

    def u_init(self, x_global):
        # x_global = x_global - self._spatial.x_left()
        # value = np.sin(x_global * np.pi * 2 / self._spatial.length())
        while x_global > self._x_right:
            x_global -= self._length
        while x_global < self._x_left:
            x_global += self._length
        quad_length = self._length / 4
        gauss_width = self._delta_x / 2
        value = (np.exp(-((x_global - quad_length) / gauss_width)**2 / 2)
            * np.sin(self._kappa_low * x_global))
        value += (np.exp(-((x_global + quad_length) / gauss_width)**2 / 2)
            * np.sin(self._kappa_high * x_global))
        return value

    def u_exact(self, x_global, t_curr):
        return self.u_init(x_global - self._a_const * t_curr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python3 wave_packet.py',
        description = 'What the program does',
        epilog = 'Text at the bottom of help')
    parser.add_argument('-n', '--n_element',
        default=10, type=int,
        help='number of elements')
    parser.add_argument('-l', '--x_left',
        default=-10.0, type=float,
        help='coordinate of the left end of the domain')
    parser.add_argument('-r', '--x_right',
        default=+10.0, type=float,
        help='coordinate of the right end of the domain')
    parser.add_argument('-o', '--rk_order',
        default=3, type=int,
        help='order of Runge--Kutta scheme')
    parser.add_argument('-b', '--t_begin',
        default=0.0, type=float,
        help='time to start')
    parser.add_argument('-e', '--t_end',
        default=10.0, type=float,
        help='time to stop')
    parser.add_argument('-s', '--n_step',
        default=100, type=int,
        help='number of time steps')
    parser.add_argument('-d', '--degree',
        default=2, type=int,
        help='degree of polynomials for approximation')
    parser.add_argument('-a', '--phase_speed',
        default=1.0, type=float,
        help='phase speed of the wave')
    parser.add_argument('--k_low',
        default=1.0, type=float,
        help='the lower wave number')
    parser.add_argument('--k_high',
        default=4.0, type=float,
        help='the higher wave number')
    args = parser.parse_args()
    print(args)
    solver = LinearAdvection(args.phase_speed, args.k_low, args.k_high,
        args.degree, args.n_element, args.x_left, args.x_right,
        ode_solver=temporal.RungeKutta(args.rk_order))
    solver.animate(args.t_begin, args.t_end, args.n_step)
