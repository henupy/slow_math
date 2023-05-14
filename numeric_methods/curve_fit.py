"""
Curve fitting with the least squares method

Sauce: https://en.wikipedia.org/wiki/Least_squares
"""

import numpy as np
import matplotlib.pyplot as plt

from typing import Callable
from inspect import signature

# For typing
num = int | float


def load_data(fname: str) -> np.ndarray:
    """
    :param fname:
    :return:
    """
    with open(fname, 'r') as f:
        lines = f.readlines()
        data = np.zeros((len(lines), 2))
        for i, line in enumerate(lines):
            line = line.strip().split()
            x = float(line[0])
            y = float(line[1])
            data[i] = np.array([x, y])

    return data


def gen_noicy_data(func: Callable, x: np.ndarray, params: np.ndarray,
                   n_points: int, noice_factor: float) -> np.ndarray:
    """
    Generates noicy data according to the given function
    :param func: Function that the datapoints follow
    :param x: Array of x-coordinates
    :param params: The parameters to the given function
    :param n_points: Number of datapoints
    :param noice_factor: How much the datapoints can deviate from the function's
        curve
    :return:
    """
    xmin, xmax = x[0], x[-1]
    rand_xs = np.random.random(n_points) * (xmax - xmin) + xmin
    rand_xs = np.array(sorted(rand_xs))
    noice_min, noice_max = 1 - noice_factor, 1 + noice_factor
    noice_arr = np.random.random(n_points) * (noice_max - noice_min) + noice_min
    y = func(rand_xs, *params) * noice_arr
    return np.array([rand_xs, y])


def _residual(func: Callable, x: np.ndarray, y: np.ndarray,
              params: np.ndarray) -> np.ndarray:
    """
    Calculates the residual r == y - f(x, params)
    :param func:
    :param x:
    :param y:
    :return:
    """
    return y - func(x, *params).T


def _gradient(func: Callable, x: np.ndarray, y: np.ndarray, params: np.ndarray,
              h: float = 0.0001) -> np.ndarray:
    """
    Calculates the gradient of the given function in all the x-coordinates
    :param func:
    :param x:
    :param params:
    :return:
    """
    n = params.shape[0]
    h_arr = np.zeros((n, n))
    new_params = np.zeros((n, n))
    for i in range(n):
        h_arr[i][i] = h
        new_params[i] = params + h_arr[i]
    grads = np.zeros((n, x.shape[0]))
    for i in range(n):
        res1 = _residual(func=func, x=x, y=y, params=new_params[i])
        res2 = _residual(func=func, x=x, y=y, params=params)
        grads[i] = (res1 - res2) / h
    return grads.T


def _gauss_newton(func: Callable, x: np.ndarray, y: np.ndarray,
                  params: np.ndarray, limit: float) -> np.ndarray:
    """
    Find the optimal parameters with the Gauss-Newton method
    :param func:
    :param x:
    :param y:
    :param params:
    :param limit:
    :return:
    """
    jacobian = _gradient(func=func, x=x, y=y, params=params)
    params_old = params
    params_new = np.zeros(params.shape)
    l2norm = 1
    while l2norm > limit:
        a = jacobian.T @ jacobian
        b = -jacobian.T @ _residual(func=func, x=x, y=y, params=params_old).T
        diff = np.linalg.solve(a, b)
        params_new = params_old + diff
        l2norm = np.sqrt(np.sum(np.power(diff, 2)))
        params_old = params_new
    return params_new


def fit_curve(fit_func: Callable, x: np.ndarray, y: np.ndarray,
              limit: float = 1e-6) -> np.ndarray:
    """
    Fits the given function to the given data using the least squares method
    :param fit_func: The function to be fitted
    :param x: x-data
    :param y: y-data
    :param limit:
    :return: Parameters for fit_func that yield the best fit with the data
    """
    n_params = len(signature(fit_func).parameters) - 1
    params = np.random.random(n_params).T
    return _gauss_newton(func=fit_func, x=x, y=y, params=params, limit=limit)


def linear(x: np.ndarray, a: num, b: num) -> np.ndarray:
    """
    Function for a line of form y == ax + b
    :param x: X-coordinates
    :param a:
    :param b:
    :return:
    """
    return a * x + b


def polynomial(x: np.ndarray, a: num, b: num, c: num) -> np.ndarray:
    """
    :param x:
    :param a:
    :param b:
    :param c:
    :return:
    """
    return a * np.power(x, 2) + b * x + c


def main() -> None:
    datafile = 'data64.txt'
    data = load_data(datafile)
    x, y = data.T[0], data.T[1]
    plt.scatter(x, y, c='r', label='Data points')
    fun = polynomial
    popt = fit_curve(fit_func=fun, x=x, y=y)
    plt.plot(x, fun(x, *popt), 'g-', label='Fitted line')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
