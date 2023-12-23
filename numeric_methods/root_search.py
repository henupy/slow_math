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


def split(f: callable, neg_guess: int | float,
          pos_guess: int | float, limit: float) -> float:
    """
    Finds the root by taking one negative and one positive guesses,
    that should be located around the root, and closes in on the root
    via some kind of binary search
    :param f: The function for which the root is to be found
    :param neg_guess:
    :param pos_guess:
    :param limit:
    :return:
    """
    c = 0
    while abs(pos_guess - neg_guess) > limit:
        c = (neg_guess + pos_guess) / 2
        if _sign(f(c)) == _sign(f(neg_guess)):
            neg_guess = c
        else:
            pos_guess = c

    return c


def polyfun(x: int | float) -> float:
    return 5 * math.atan(x) - x + 1


def main():
    guess, limit = 0, 0.0001
    neg, pos = -1, 1
    diff = middle
    root_newton = newton(polyfun, diff, guess, limit)
    root_split = split(polyfun, neg, pos, limit)
    print(f"Root (newton): {root_newton}")
    print(f"Root (split): {root_split}")


if __name__ == "__main__":
    main()
