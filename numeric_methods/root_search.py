"""
Couple of ways to find the root(s) (x-intercepts) of a function
"""

import math

from numeric_derivation import middle


def newton(f: callable, diff: callable, a: int | float, limit: float) -> float:
    """
    Finds the root using Newton's method
    :param f: The function for which the root is to be found
    :param diff: The derivative of the function
    :param a: Initial guess for the root
    :param limit: Approximation limit; how close should the
        approximation be before the iteration is ended
    :return:
    """
    error = int(1e10)
    while abs(error) > limit:
        error = f(a) / diff(f, a)
        a -= error
    return a


def _sign(num: int | float) -> int:
    """
    Returns 1 if the sign of "num" is positive or zero, else -1
    :param num:
    :return:
    """
    if num >= 0:
        return 1
    return -1


def bisect(f: callable, a: int | float, b: int | float, limit: float) -> float:
    """
    Finds the root by taking one negative and one positive guesses,
    that should be located around the root, and closes in on the root
    via some kind of binary search
    :param f: The function for which the root is to be found
    :param a:
    :param b:
    :param limit:
    :return:
    """
    if a >= b:
        raise ValueError("a must be smaller than b")
    aval = f(a)
    bval = f(b)
    neg = min(aval, bval)
    pos = max(aval, bval)
    if not (neg < 0 and pos > 0):
        msg = "f must have a negative value with one guess and positive with the other"
        raise ValueError(msg)
    iters = 10000
    c, mid = 0, 0
    while abs(a - b) > limit and c <= iters:
        mid = (a + b) / 2
        if _sign(f(mid)) == _sign(f(a)):
            a = mid
        else:
            b = mid
        c += 1
    return mid


def secant(f: callable, x0: int | float, x1: int | float, limit: float) -> float:
    """
    Finds the root of f using the Secant method
    (https://en.wikipedia.org/wiki/Secant_method)
    :param f:
    :param x0:
    :param x1:
    :param limit:
    :return:
    """
    c, iters = 0, 1e5
    err = abs(x1 - x0)
    while err > limit and c < iters:
        fx1 = f(x1)
        x = x1 - fx1 * (x1 - x0) / (fx1 - f(x0))
        x0 = x1
        x1 = x
        err = abs(x1 - x0)
        c += 1
    return x1


def foo(x: int | float) -> float:
    return 5 * math.atan(x) - x + 1


def bar(x: int | float) -> float:
    return x * x - 612


def main():
    guess, limit = 20, 1e-4
    neg, pos = 10, 30
    diff = middle
    fun = bar
    root_newton = newton(fun, diff, guess, limit)
    root_bisect = bisect(fun, neg, pos, limit)
    root_secant = secant(fun, neg, pos, limit)
    print(f"Root (newton): {root_newton}")
    print(f"Root (bisect): {root_bisect}")
    print(f"Root (secant): {root_secant}")


if __name__ == "__main__":
    main()
