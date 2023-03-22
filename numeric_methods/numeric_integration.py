"""
Some ways to integrate numerically a one-dimensional function, i.e., find
the area under a curve of form y = f(x)
"""

import numpy as np
import matplotlib.pyplot as plt

from typing import Callable

# For typing
numeric = int | float


def mc(f: Callable, a: numeric, b: numeric, num_points: int = int(1e5)) -> numeric:
    """
    Numerical integration using the Monte Carlo method, i.e., sampling
    a number of random points
    :param f: Function to be integrated
    :param a: Lower integration limit
    :param b: Upper integration limit
    :param num_points: Number of random points to be sampled (default 1e5)
    :return:
    """
    x = np.linspace(a, b, num_points, endpoint=True)
    y = f(x)
    y_max = np.max(y)
    y_points = np.random.random(size=(num_points, )) * y_max
    area_fraction = np.sum(y_points < y) / num_points
    return area_fraction * (b - a) * y_max


def rect(f: Callable, a: numeric, b: numeric, n_rects: int = int(1e3)) -> numeric:
    """
    Calculates the integral by dividing the area under the curve
    to small rectangles, and summing the area of the rectangles
    :param f: Function to be integrated
    :param a: Lower integration limit
    :param b: Upper integration limit
    :param n_rects: Number of rectangles that the area is divided into
    :return:
    """
    x = np.linspace(a, b, n_rects, endpoint=True)
    y = f(x)
    dx = x[1] - x[0]
    return float(np.sum(y * dx))


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


def expfun(x: numeric | np.ndarray) -> float | np.ndarray:
    """
    :param x:
    :return:
    """
    return np.power(x, 3) * np.exp(np.power(x, 2))


def fun1(x: int | float | np.ndarray) -> float | np.ndarray:
    """
    :param x:
    :return:
    """
    return np.power(x, 2) + 2 * x


def fun2(x: int | float | np.ndarray) -> float | np.ndarray:
    """
    :param x:
    :return:
    """
    return x - 2


def main():
    # TODO: Add functionality for 'negative' areas (Monte Carlo)
    # TODO: Add Simpson's rule
    a, b = 0, 4
    func = fun1
    area_mc = mc(f=func, a=a, b=b)
    area_rect = rect(f=func, a=a, b=b)
    print(f'Monte Carlo: {area_mc:.4f}')
    print(f'Rectangle: {area_rect:.4f}')
    x = np.linspace(0, 5, 100)
    y = func(x)
    plt.plot(x, y)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
