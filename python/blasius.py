"""Generate the Blasius solution for the flat plate boundary layer.
"""
import argparse
import numpy as np
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt


class Interpolation:
    _data = np.loadtxt('blasius.csv', delimiter=',', skiprows=1)
    _eta = _data[:, 0]
    _u_data = _data[:, 1]
    _v_data = _data[:, 2]
    _u_spline = CubicSpline(_eta, _u_data, bc_type='not-a-knot')
    _v_spline = CubicSpline(_eta, _v_data, bc_type='not-a-knot')

    def __init__(self, u_infty, nu) -> None:
        self._u_infty = u_infty
        self._nu = nu
        self._nu_over_u_infty = nu / u_infty

    def get_eta(self, x, y) -> float:
        return y / np.sqrt(x * self._nu_over_u_infty)

    def get_dimensionless_u(self, x, y) -> float:
        eta = self.get_eta(x, y)
        return self._u_spline(eta)

    def get_dimensional_u(self, x, y):
        return self.get_dimensionless_u(x, y) * self._u_infty

    def get_dimensionless_v(self, x, y) -> float:
        eta = self.get_eta(x, y)
        reynolds_x = x / self._nu_over_u_infty
        return self._v_spline(eta) / np.sqrt(reynolds_x) / 2

    def get_dimensional_v(self, x, y):
        return self.get_dimensionless_v(x, y) * self._u_infty


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python blasius.py',
        description = 'What the program does',
        epilog = 'Text at the bottom of help')
    parser.add_argument('-U', '--u_infty',
        default=68.3, type=float,
        help='free stream velocity')
    parser.add_argument('-nu', '--viscosity',
        default=1.57e-5, type=float,
        help='kinematic viscosity')
    parser.add_argument('-x', '--x_query',
        default=0.1, type=float,
        help='position of the query section')
    parser.add_argument('-y', '--y_max',
        default=0.001, type=float,
        help='upper bound of the query range')
    parser.add_argument('-n', '--n_sample',
        default=100, type=int,
        help='number of sample points on the section')
    parser.add_argument('--write_csv',
        action='store_true')
    args = parser.parse_args()
    print(args)
    u_infty = args.u_infty
    nu = args.viscosity
    solution = Interpolation(u_infty, nu)
    x = args.x_query
    y = np.linspace(0, args.y_max, args.n_sample)
    eta = solution.get_eta(x, y)
    u = solution.get_dimensional_u(x, y)
    v = solution.get_dimensional_v(x, y)

    if args.write_csv:
        np.savetxt(f'blasius_x={x}.csv', np.array([u, v, y]).T, delimiter=',',
            header='u,v,y', comments='')

    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(u / u_infty, eta)
    plt.xlabel(r'$u/u_\infty$')
    plt.ylabel(r'$\eta = y/\sqrt{\nu x/u_\infty}$')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(v, y / args.y_max)
    plt.xlabel(r'$v$')
    plt.ylabel(f'{r"$y$"}/{args.y_max}')
    plt.grid()
    plt.tight_layout()
    fig.savefig('Blasius.svg')
