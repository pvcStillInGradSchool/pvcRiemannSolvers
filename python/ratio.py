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


y_0 = 0.0
y_n = 0.02
dy_0 = 1e-5
n = 64
a = get_ratio(y_0, y_n, dy_0, n)
print(f'a = {a}')

powers = np.arange(0, n + 1)
y = y_0 + (a**powers - 1) / (a - 1) * dy_0
y_n = y[-1]
print(f'y_0 = {y_0}')
print(f'y_n = {y_n}')
