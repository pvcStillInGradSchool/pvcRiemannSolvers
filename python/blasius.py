"""Generate the Blasius solution for the flat plate boundary layer.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt


class Interpolation:
    _data = np.loadtxt('blasius.csv', delimiter=',', skiprows=1)
    _eta = _data[:, 0]
    _df = _data[:, 1]
    _eta_times_df_minus_f = _data[:, 2]
    _df_spline = CubicSpline(_eta, _df, bc_type='not-a-knot')
    _eta_times_df_minus_f_spline = CubicSpline(_eta, _data[:, 2])

    def __init__(self, u_infty, nu) -> None:
        self._u_infty = u_infty
        self._nu = nu
        self._nu_over_u_infty = nu / u_infty

    def get_eta(self, x, y) -> float:
        return y / np.sqrt(x * self._nu_over_u_infty)

    def get_dimensionless_u(self, x, y) -> float:
        eta = self.get_eta(x, y)
        return self._df_spline(eta)

    def get_dimensional_u(self, x, y):
        return self.get_dimensionless_u(x, y) * self._u_infty

    def get_dimensionless_v(self, x, y) -> float:
        eta = self.get_eta(x, y)
        reynolds = x / self._nu_over_u_infty
        return self._eta_times_df_minus_f_spline(eta) / np.sqrt(reynolds) / 2

    def get_dimensional_v(self, x, y):
        return self.get_dimensionless_v(x, y) * self._u_infty


if __name__ == '__main__':
    u_infty = 68.3
    nu = 1.57e-5
    solution = Interpolation(u_infty, nu)
    x = 0.1
    y = np.linspace(0, 0.001, 101)
    eta = solution.get_eta(x, y)
    u = solution.get_dimensional_u(x, y)
    v = solution.get_dimensional_v(x, y)

    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(u / u_infty, eta)
    plt.xlabel(r'$u/u_\infty$')
    plt.ylabel(r'$\eta$')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(v, eta)
    plt.xlabel(r'$v$')
    plt.ylabel(r'$\eta$')
    plt.grid()
    plt.tight_layout()
    fig.savefig('Blasius.svg')
