import gmsh
import sys
import numpy as np
from scipy import optimize as opt


def get_ratio(y_0: float, y_n: float, dy_0: float, n: int):
    """Get the value of `a` such that `dy_0 == y[1] - y[0]` and `a == (y[i+1] - y[i]) / (y[i] - y[i-1])` for all `i` in `range(0, n + 1)`.
    """
    def f(a: float):
        val = (a**n - 1) / (a - 1) - ((y_n - y_0) / dy_0)
        return val
    def df(a: float):
        return n * a**(n - 1) / (a - 1) - (a**n - 1) / (a - 1)
    return opt.newton(f, x0=1.01, fprime=df, tol=1e-15)


def get_first_layer(y_0: float, y_n: float, a: float, n: int):
    """Get the value of `dy_0 == y[1] - y[0]` such that `a == (y[i+1] - y[i]) / (y[i] - y[i-1])` for all `i` in `range(0, n + 1)`.
    """
    return (y_n - y_0) * (a - 1) / (a**n - 1)


def demo_gmsh(y_0: float, y_n: float, dy_0: float, n: int):
    gmsh.initialize(sys.argv)

    gmsh.model.geo.addPoint(y_0, 0, 0)
    gmsh.model.geo.addPoint(y_n, 0, 0)
    gmsh.model.geo.addLine(1, 2)
    gmsh.model.geo.synchronize()

    ratio = get_ratio(y_0, y_n, dy_0, n)
    gmsh.model.mesh.setTransfiniteCurve(1, n + 1, "Progression", ratio)
    gmsh.model.mesh.generate(1)
    gmsh.model.geo.synchronize()
    nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes(1, 1, True)
    nodeXs = nodeCoords[0:len(nodeCoords):3]
    nodeYs = nodeCoords[1:len(nodeCoords):3]
    nodeZs = nodeCoords[2:len(nodeCoords):3]
    def shift(a: np.ndarray):
        """Convert
            inner nodes | left end | right end
          to
            left end | inner nodes | right end
        """
        n = len(a)
        left_val = a[-2]
        for i in range(n - 2, 0, -1):
            a[i] = a[i - 1]
        a[0] = left_val
    shift(nodeTags)
    shift(nodeXs)
    shift(nodeYs)
    shift(nodeZs)
    shift(nodeParams)
    for i in range(len(nodeTags)):
        print(i, nodeTags[i], nodeParams[i], nodeXs[i], nodeYs[i], nodeZs[i])
        if i > 1:
            param_ratio = (nodeParams[i] - nodeParams[i - 1]) / (nodeParams[i - 1] - nodeParams[i - 2])
            assert np.abs(ratio - param_ratio) < 1e-6
            x_ratio = (nodeXs[i] - nodeXs[i - 1]) / (nodeXs[i - 1] - nodeXs[i - 2])
            assert np.abs(ratio - x_ratio) < 1e-6

    gmsh.finalize()


y_0 = 0.0
y_n = 0.02
dy_0 = 1e-5
n = 64
a = get_ratio(y_0, y_n, dy_0, n)
print(f'a = {a}')
print(dy_0, get_first_layer(y_0, y_n, a, n))

powers = np.arange(0, n + 1)
y = y_0 + (a**powers - 1) / (a - 1) * dy_0
y_n = y[-1]
print(f'y_0 = {y_0}')
print(f'y_n = {y_n}')

demo_gmsh(y_0, y_n, dy_0, n)
