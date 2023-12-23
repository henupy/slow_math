"""
Some methods to differentiate numerically
"""

import math


def backward(f: callable, x: int | float,
             h: int | float = 0.00001) -> float:
    """
    :param f: Function to be differentiated, must be a python-function
    :param x: The point where the function's derivative shall be
        estimated
    :param h: The small value that is added to the function so that
        its change can be calculated
    """
    return (f(x) - f(x - h)) / h


def forward(f: callable, x: int | float,
            h: int | float = 0.00001) -> float:
    """
    :param f: Function to be differentiated, must be a python-function
    :param x: The point where the function's derivative shall be
        estimated
    :param h: The small value that is added to the function so that
        its change can be calculated
    """
    return (f(x + h) - f(x)) / h


def middle(f: callable, x: int | float,
           h: int | float = 0.00001) -> float:
    """
    :param f: Function to be differentiated, must be a python-function
    :param x: The point where the function's derivative shall be
        estimated
    :param h: The small value that is added to the function so that
        its change can be calculated
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def expfun(x: int | float) -> float:
    return 5 * math.exp(-0.1 * x)


def accuraat(x: int | float) -> float:
    return -0.5 * math.exp(-0.1 * x)


def main():
    x = 0
    b = backward(expfun, x)
    f = forward(expfun, x)
    m = middle(expfun, x)
    acc = accuraat(x)
    print(f"Back: {b}, forward: {f}, middle: {m}, accurate: {acc}")
    print(f"Errors: Back: {abs(b - acc)}, forward: {abs(f - acc)} "
          f"Middle: {abs(m - acc)}")


if __name__ == "__main__":
    main()
