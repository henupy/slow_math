"""
Line integral in a 2d vector field

Some examples are from Khan Academy: https://www.khanacademy.org/math/
multivariable-calculus/integrating-multivariable-functions/
line-integrals-in-vector-fields-articles/a/line-integrals-in-a-vector-field
"""

import numpy as np

from typing import Callable


def curve(s: np.ndarray) -> np.ndarray:
    """
    Example of a curve from Khan Academy
    :param s:
    :return:
    """
    x = 100 * (s - np.sin(s))
    y = 100 * (-s - np.sin(s))
    return np.array([x, y])


def circle(s: int | float | np.ndarray) -> np.ndarray:
    """
    A circle centeread around the point (2, 0) (from Khan Academy)
    :param s:
    :return:
    """
    return np.array([np.cos(s) + 2, np.sin(s)])


def grav_field(x: int | float, y: int | float) -> np.ndarray:
    """
    Gravity as a vector field (from Khan Academy)
    :param x:
    :param y:
    :return:
    """
    _, _ = x, y  # These aren't needed in this case
    m = 170e3  # Mass of the object [kg]
    g = -9.81  # Gravitational acceleration [m/s]
    return np.array([0, m * g])


def tornado(x: int | float, y: int | float) -> np.ndarray:
    """
    Rotating vector field (from Khan Academy)
    :param x:
    :param y:
    :return:
    """
    return np.array([-y, x])


def line_int(s: np.ndarray, line: Callable, field: Callable,
             eps: float = 1e-4) -> int | float:
    """
    Line integral in a 2d vector field
    :param s:
    :param line:
    :param field:
    :param eps:
    :return:
    """
    # Derivative of the line at each point in s
    s0, s1 = s - eps, s + eps
    dlineds = ((line(s1) - line(s0)) / (2 * eps)).T
    # Replace the integral with a sum
    ds = s[1] - s[0]
    val = 0
    for i in range(s.shape[0]):
        x, y = line(s[i])
        val += np.dot(field(x, y), (dlineds[i] * ds))
    return val


def main():
    # The curve defined in terms of time
    start, end, dt = 0, 2 * np.pi, 0.001  # [s]
    s = np.linspace(start, end, int((end - start) / dt))  # Time range [s]
    line = circle
    field = tornado
    lineint = line_int(s, line, field)
    print(lineint)


if __name__ == "__main__":
    main()
