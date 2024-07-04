import numpy as np
from matplotlib_wrapper import savefig
from matplotlib import pyplot as plt


def dist_to_vertices(point: np.ndarray, polygon: np.ndarray, p: int, reduce: callable):
    n_vertex = len(polygon)
    distances = np.ndarray(n_vertex)
    for i in range(n_vertex):
        diff = np.abs(polygon[i] - point)
        if p == np.inf:
            distances[i] = np.max(diff)
        else:
            p_sum = np.sum(diff**p)
            distances[i] = np.power(p_sum, 1 / p)
    return reduce(distances)


def dist_to_edges(point: np.ndarray, polygon: np.ndarray, reduce: callable):
    n_vertex = len(polygon)
    distances = np.ndarray(n_vertex)
    for i in range(n_vertex):
        a = polygon[i]
        b = polygon[i - 1]
        pa = a - point
        ab = a - b
        normal = np.array([ab[1], -ab[0]])
        normal /= np.linalg.norm(normal)
        pn_len = np.abs(normal.dot(pa))
        distances[i] = pn_len
    return reduce(distances)


def contour(prefix: str, polygon: np.ndarray, distance: callable):
    step = 0.05
    a = 5
    x = np.arange(-a, a, step)
    y = np.arange(-a, a, step)
    X, Y = np.meshgrid(x, y)
    Z = np.ndarray(X.shape)
    for i in range(len(y)):
        for j in range(len(x)):
            point = np.array([X[i][j], Y[i][j]])
            Z[i][j] = distance(point, polygon)
    plt.figure()
    c = plt.contourf(X, Y, Z)
    for i in range(len(polygon)):
        a, b = polygon[i], polygon[(i+1)%len(polygon)]
        plt.plot([a[0], b[0]], [a[1], b[1]], 'r-')
    plt.colorbar(c)
    plt.axis('equal')
    plt.tight_layout()
    savefig(f'{prefix}')


if __name__ == '__main__':
    polygon = np.array([
        [-1, -1], [1, -1], [1, 1], [-1, 1]
    ], dtype=float)
    for p in (1, 2, 3, np.inf):
        contour(f'quadrangle_sum_p={p}', polygon,
            lambda point, polygon: dist_to_vertices(point, polygon, p, np.sum))
        contour(f'quadrangle_min_p={p}', polygon,
            lambda point, polygon: dist_to_vertices(point, polygon, p, np.min))
        contour(f'quadrangle_sum_edge', polygon,
            lambda point, polygon: dist_to_edges(point, polygon, np.sum))
        contour(f'quadrangle_min_edge', polygon,
            lambda point, polygon: dist_to_edges(point, polygon, np.min))
    polygon = np.array([
        [-1, -1], [1, -1], [0, 1]
    ], dtype=float)
    for p in (1, 2, 3, np.inf):
        contour(f'triangle_sum_p={p}', polygon,
            lambda point, polygon: dist_to_vertices(point, polygon, p, np.sum))
        contour(f'triangle_min_p={p}', polygon,
            lambda point, polygon: dist_to_vertices(point, polygon, p, np.min))
        contour(f'triangle_sum_edge', polygon,
            lambda point, polygon: dist_to_edges(point, polygon, np.sum))
        contour(f'triangle_min_edge', polygon,
            lambda point, polygon: dist_to_edges(point, polygon, np.min))
