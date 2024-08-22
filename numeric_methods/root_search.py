"""
Couple of ways to find the root(s) (x-intercepts) of a function
"""

import math

from numeric_derivation import middle


def newton(f: callable, diff: callable, a: int | float,
           limit: float) -> float:
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
    if not neg < 0 and pos > 0:
        msg = "f must have a negative value with one guess and positive with the other"
        raise ValueError(msg)
    iters = 10000
    c, mid = 0, 0
    while abs(a - b) > limit and c < iters:
        mid = (a + b) / 2
        if _sign(f(mid)) == _sign(f(a)):
            a = mid
        else:
            b = mid
        c += 1
    return mid


def foo(x: int | float) -> float:
    return 5 * math.atan(x) - x + 1


def main():
    guess, limit = 0, 0.0001
    neg, pos = -1, 1
    diff = middle
    root_newton = newton(foo, diff, guess, limit)
    root_bisect = bisect(foo, neg, pos, limit)
    print(f"Root (newton): {root_newton}")
    print(f"Root (bisect): {root_bisect}")


if __name__ == "__main__":
    main()
