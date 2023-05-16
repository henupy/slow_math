"""
Directional derivative in a 2d scalar field

Some examples are from Wikipedia: https://en.wikipedia.org/wiki/Directional_derivative
"""

import numpy as np
import matplotlib.pyplot as plt

from gradient import grad
from typing import Callable


def sqroot(x: int | float | np.ndarray, y: int | float | np.ndarray) \
        -> int | float | np.ndarray:
    """
    Example 2d function
    :param x:
    :param y:
    :return:
    """
    return np.sqrt(np.power(x, 2) + np.power(y, 2))


def x2y2(x: int | float | np.ndarray, y: int | float | np.ndarray) \
        -> int | float | np.ndarray:
    """
    Example 2d function from Wikipedia
    :param x:
    :param y:
    :return:
    """
    return np.power(x, 2) + np.power(y, 2)


def dirdev(x: int | float, y: int | float, fun: Callable, v: np.ndarray,
           eps: float = 1e-4) -> np.ndarray:
    """
    Directional derivative of the given 2d function at the location (x, y)
    in the direction of v
    :param x:
    :param y:
    :param fun:
    :param v:
    :param eps:
    :return:
    """
    # Assert v is a normal vector
    v = v / np.linalg.norm(v)
    x0, y0 = x - eps * v[0], y - eps * v[1]
    x1, y1 = x + eps * v[0], y + eps * v[1]
    return (fun(x1, y1) - fun(x0, y0)) / (2 * eps)


def visualise(x: int | float, y: int | float, fun: Callable, v: np.ndarray,
              eps: float = 1e-4) -> None:
    """
    Plots the scalar field, the gradient vector at (x, y), and the vector v
    scaled by the directional derivative at (x, y) in the direction of v
    :param x:
    :param y:
    :param fun:
    :param v:
    :param eps:
    :return:
    """
    pad = 5
    xa = np.array([x])
    ya = np.array([y])
    # TODO: check the epsilon thing
    nabla = grad(xa, ya, fun, eps)[0][0]
    v_scaled = v / np.linalg.norm(v) * dirdev(x, y, fun, v, eps)
    scaler = np.linalg.norm(nabla)
    xmin, xmax = x - scaler - pad, x + scaler + pad
    ymin, ymax = y - scaler - pad, y + scaler + pad
    xx, yy = np.linspace(xmin, xmax, 21), np.linspace(ymin, ymax, 21)
    xx, yy = np.meshgrid(xx, yy)
    field = fun(xx, yy)
    plt.contourf(xx, yy, field)
    plt.quiver(x, y, nabla[0], nabla[1], color='black', label='Gradient',
               angles='xy', scale_units='xy', scale=1)
    plt.quiver(x, y, v_scaled[0], v_scaled[1], color='red', label='Dir. Derivative',
               angles='xy', scale_units='xy', scale=1)
    plt.legend()
    plt.show()


def main():
    x, y = -5, 5
    v = np.array([1, -4])
    fun = x2y2
    visualise(x, y, fun, v)


if __name__ == '__main__':
    main()
