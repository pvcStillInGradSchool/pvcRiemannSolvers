import numpy as np
from matplotlib_wrapper import savefig
from matplotlib import pyplot as plt


class Quadrangle4():
    """Coordinate map on 4-node quadrilateral elements.
    """

    def __init__(self, a, b, c, d):
        self._global_coords = np.ndarray(shape=(2, 4))
        self._global_coords[:, 0] = a
        self._global_coords[:, 1] = b
        self._global_coords[:, 2] = c
        self._global_coords[:, 3] = d

    def local_to_shape(self, local):
        x, y = local[0], local[1]
        shape = np.array([
          (1 - x) * (1 - y) / 4,
          (1 + x) * (1 - y) / 4,
          (1 + x) * (1 + y) / 4,
          (1 - x) * (1 + y) / 4,
        ])
        return shape

    def local_to_global(self, local):
        shape = self.local_to_shape(local)
        return self._global_coords.dot(shape)

    def plot(self):
        plt.plot(self._global_coords[0,:], self._global_coords[1,:], 'b-')
        x = (self._global_coords[0,-1], self._global_coords[0,0])
        y = (self._global_coords[1,-1], self._global_coords[1,0])
        plt.plot(x, y, 'b-')

    def plot_history(self, local_history):
        global_prev = self.local_to_global(local_history[0])
        for local in local_history:
            global_curr = self.local_to_global(local)
            x = (global_prev[0], global_curr[0])
            y = (global_prev[1], global_curr[1])
            plt.plot(x, y, 'g+-')
            global_prev = global_curr

    def plot_grid(self, xi_range, eta_range):
        rgba = np.linspace(0.1, 0.9, len(xi_range))
        i = 0
        xi_str = r'$\xi$'
        for xi in xi_range:
            head = self.local_to_global((xi, -10))
            tail = self.local_to_global((xi, +10))
            x = (head[0], tail[0])
            y = (head[-1], tail[-1])
            if i % 2:
                plt.plot(x, y, color=('c', rgba[i]), linestyle='--')
            else:
                plt.plot(x, y, color=('c', rgba[i]), label=f'{xi_str} = {xi}')
            i += 1
        i = 0
        eta_str = r'$\eta$'
        for eta in eta_range:
            head = self.local_to_global((-10, eta))
            tail = self.local_to_global((+10, eta))
            x = (head[0], tail[0])
            y = (head[-1], tail[-1])
            if i % 2:
                plt.plot(x, y, color=('k', rgba[i]), linestyle='--')
            else:
                plt.plot(x, y, color=('k', rgba[i]), label=f'{eta_str} = {eta}')
            i += 1


if __name__ == '__main__':
    quad = Quadrangle4(
        (2.00000, 1.000000),
        (1.94313, 0.878607),
        (2.00493, 0.845382),
        (2.06283, 0.874438))
    quad.plot_grid(np.linspace(-5, 5, 11), np.linspace(-5, 5, 11))
    quad.plot()
    plt.plot(2.05723, 0.777978, 'rx')
    history = np.loadtxt('history.txt', delimiter=' ')
    quad.plot_history(history)
    plt.axis('equal')
    plt.xlim(1.90, 2.10)
    plt.ylim(0.75, 1.05)
    plt.legend()
    plt.tight_layout()
    savefig('quad4')

