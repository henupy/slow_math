"""
Some ways to integrate numerically a one-dimensional function, i.e., find
the area under a curve of form y = f(x)
"""

import numpy as np

from typing import Callable

# For typing
numeric = int | float


def mc(f: Callable, a: numeric, b: numeric, num_points: int = int(1e5)) -> numeric:
    """
    Numerical integration using the Monte Carlo method, i.e., sampling
    a number of random points and finding out how many land inside the area
    that the function specifies
    :param f: Function to be integrated
    :param a: Lower integration limit
    :param b: Upper integration limit
    :param num_points: Number of random points to be sampled (default 1e5)
    :return:
    """
    x = np.linspace(a, b, num_points)
    y = f(x)
    y_max = np.max(y)
    y_points = np.random.random(size=(num_points, )) * y_max
    area_fraction = np.sum(y_points < y) / num_points
    return area_fraction * (b - a) * y_max


def trapezoid(f: Callable, a: numeric, b: numeric, n_rects: int = int(1e3)) -> numeric:
    """
    Calculates the definitive integral of the given function by using the
    trapezoidal rule (https://en.wikipedia.org/wiki/Trapezoidal_rule) using a
    uniform grid
    :param f: Function to be integrated
    :param a: Lower integration limit
    :param b: Upper integration limit
    :param n_rects: Number of trapezoids that the area is divided into
    :return:
    """
    x = np.linspace(a, b, n_rects)
    y = f(x)
    dx = x[1] - x[0]
    return dx * (np.sum(y[1:-1]) + (y[-1] + y[0]) / 2)


def _simpson13(f: Callable, a: numeric, b: numeric, n: int) -> numeric:
    """
    Implementation of Simpson's 1/3 rule
    :param f:
    :param a:
    :param b:
    :param n:
    :return:
    """
    x = np.linspace(a, b, n + 1)
    y = f(x)
    dx = x[1] - x[0]
    odds_sum = np.sum(y[1:-1:2])
    evens_sum = np.sum(y[2:-1:2])
    return 1 / 3 * dx * (y[0] + 4 * odds_sum + 2 * evens_sum + y[-1])


def _simpson38(f: Callable, a: numeric, b: numeric, n: int) -> numeric:
    """
    Implementation of Simpson's 3/8 rule
    :param f:
    :param a:
    :param b:
    :param n:
    :return:
    """
    x = np.linspace(a, b, n + 1)
    y = f(x)
    dx = x[1] - x[0]
    non_divs = np.arange(1, x.shape[0]) % 3 != 0
    non_divs_sum = np.sum(y[1:] * non_divs)
    divs_sum = np.sum(y[3:-1:3])
    return 3 / 8 * dx * (y[0] + 3 * non_divs_sum + 2 * divs_sum + y[-1])


def simpson(f: Callable, a: numeric, b: numeric, n: int, rule: str = '1/3') -> numeric:
    """
    Calculates the definitive integral of the given function using the Simpson's
    composite rule (https://en.wikipedia.org/wiki/Simpson%27s_rule). The specific
    method/rule can be either Simpson's 1/3 rule or Simpson's 3/8 rule.
    :param f: The function to be integrated
    :param a: Lower integration limit
    :param b: Upper integration limit
    :param n: Number of areas that the interval (a ... b) is split into. For the '1/3'
    rule n must be even, and for the '3/8' rule n must be divisible by three. If an
    'unsuitable' n is given, it will be modified to be suitable.
    :param rule: Which rule ('1/3' or ('3/8') is used. Defaults to ('1/3')
    :return:
    """
    if rule == '3/8':
        if n % 3 != 0:
            n -= n % 3
        return _simpson38(f=f, a=a, b=b, n=n)
    if rule != '1/3':
        print("Unrecognized rule, used the default rule '1/3'")
    if n % 2 != 0:
        n += 1
    return _simpson13(f=f, a=a, b=b, n=n)


def logfun(x: numeric | np.ndarray) -> numeric | np.ndarray:
    """
    :param x:
    :return:
    """
    return np.log(x) / x


def cosfun(x: numeric | np.ndarray) -> float | np.ndarray:
    """
    :param x:
    :return:
    """
    return x * np.cos(x)


def fun1(x: numeric | np.ndarray) -> numeric | np.ndarray:
    """
    :param x:
    :return:
    """
    return .5 * np.power(x, 2) - 2 * x


def fun2(x: numeric | np.ndarray) -> numeric | np.ndarray:
    """
    :param x:
    :return:
    """
    return x - 2


def fun3(t: numeric | np.ndarray) -> numeric | np.ndarray:
    """
    :param t:
    :return:
    """
    a = 5587  # [m^3/d]
    b = -1421  # [m^3/d^2]
    c = 125  # [m^3/d^3]
    d = -2.85  # [m^3/d^3]
    t2 = np.power(t, 2)
    t3 = np.power(t, 3)
    return d * t3 + c * t2 + b * t + a


def main():
    # TODO: Add functionality for 'negative' areas (Monte Carlo)
    # TODO: Add Romberg's method
    # TODO: Add the midpoint rule
    a, b = 1, 25
    n_rects = 1000
    simp_rule = '1/3'
    func = fun3
    area_mc = mc(f=func, a=a, b=b)
    area_rect = trapezoid(f=func, a=a, b=b, n_rects=n_rects)
    area_simp = simpson(f=func, a=a, b=b, n=n_rects, rule=simp_rule)
    print(f'Monte Carlo: {area_mc:.4f}')
    print(f'Rectangle: {area_rect:.4f}')
    print(f"Simpson's rule (rule {simp_rule}): {area_simp:.4f}")


if __name__ == '__main__':
    main()
