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

    def __init__(self, u_infty, nu) -> None:
        self._u_infty = u_infty
        self._nu = nu
        self._nu_over_u_infty = nu / u_infty

    def get_eta(self, x, y) -> float:
        return y / np.sqrt(x * self._nu_over_u_infty)

    def get_u(self, x, y) -> float:
        eta = self.get_eta(x, y)
        print(eta)
        return self._u_infty * self._df_spline(eta)


if __name__ == '__main__':
    u_infty = 68.3
    nu = 1.57e-5
    solution = Interpolation(u_infty, nu)
    x = 0.1
    y = np.linspace(0, 0.001, 101)
    print(y)
    eta = solution.get_eta(x, y)
    u = solution.get_u(x, y)

    fig = plt.figure()
    plt.plot(y, u, label=r'$u$')
    plt.xlabel(r'$y$')
    plt.legend()
    fig.savefig('Blasius.svg')
