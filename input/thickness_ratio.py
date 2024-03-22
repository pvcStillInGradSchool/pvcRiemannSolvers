import gmsh
import sys
import numpy as np
from scipy import optimize as opt


def get_thickness_ratio(y_0: float, y_n: float, dy_0: float, n_layer: int):
    """Get the value of `h_ratio` such that `dy_0 == y[1] - y[0]` and `h_ratio == (y[i+1] - y[i]) / (y[i] - y[i-1])` for all `i` in `range(0, n_layer + 1)`.
    """
    def f(h_ratio: float):
        val = (h_ratio**n_layer - 1) / (h_ratio - 1) - ((y_n - y_0) / dy_0)
        return val
    def df(h_ratio: float):
        return n_layer * h_ratio**(n_layer - 1) / (h_ratio - 1) - (h_ratio**n_layer - 1) / (h_ratio - 1)
    return opt.newton(f, x0=1.01, fprime=df, tol=1e-15)


def get_first_layer_thickness(y_0: float, y_n: float, h_ratio: float, n_layer: int):
    """Get the value of `dy_0 == y[1] - y[0]` such that `h_ratio == (y[i+1] - y[i]) / (y[i] - y[i-1])` for all `i` in `range(0, n_layer + 1)`.
    """
    return (y_n - y_0) * (h_ratio - 1) / (h_ratio**n_layer - 1)


def demo_gmsh(y_0: float, y_n: float, dy_0: float, n_layer: int):
    gmsh.initialize(sys.argv)

    gmsh.model.geo.addPoint(0, y_0, 0)
    gmsh.model.geo.addPoint(0, y_n, 0)
    gmsh.model.geo.addLine(1, 2)
    gmsh.model.geo.synchronize()

    ratio = get_thickness_ratio(y_0, y_n, dy_0, n_layer)
    gmsh.model.mesh.setTransfiniteCurve(1, n_layer + 1, "Progression", ratio)
    gmsh.model.mesh.generate(1)
    gmsh.model.geo.synchronize()
    nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes(1, 1, True)
    nodeXs = nodeCoords[0:len(nodeCoords):3]
    nodeYs = nodeCoords[1:len(nodeCoords):3]
    nodeZs = nodeCoords[2:len(nodeCoords):3]
    assert nodeXs.all() == 0
    assert nodeZs.all() == 0
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
    shift(nodeYs)
    shift(nodeParams)
    for i in range(len(nodeTags)):
        print(i, nodeTags[i], nodeParams[i], nodeYs[i])
        if i > 1:
            param_ratio = (nodeParams[i] - nodeParams[i - 1]) / (nodeParams[i - 1] - nodeParams[i - 2])
            assert np.abs(ratio - param_ratio) < 1e-6
            y_ratio = (nodeYs[i] - nodeYs[i - 1]) / (nodeYs[i - 1] - nodeYs[i - 2])
            assert np.abs(ratio - y_ratio) < 1e-6

    gmsh.finalize()


if __name__ == '__main__':
    y_0 = 0.0
    y_n = 0.02
    dy_0 = 1e-5
    n_layer = 64
    h_ratio = get_thickness_ratio(y_0, y_n, dy_0, n_layer)
    print(f'h_ratio = {h_ratio}')
    print(dy_0, get_first_layer_thickness(y_0, y_n, h_ratio, n_layer))

    powers = np.arange(0, n_layer + 1)
    y = y_0 + (h_ratio**powers - 1) / (h_ratio - 1) * dy_0
    y_n = y[-1]
    print(f'y_0 = {y_0}')
    print(f'y_n = {y_n}')

    demo_gmsh(y_0, y_n, dy_0, n_layer)
