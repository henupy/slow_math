"""
Laplace operator for a 2d scalar field
"""

import numpy as np
import matplotlib.pyplot as plt

from gradient import grad
from divergence import div

# For typing
num = int | float


def sqroot(x: num | np.ndarray, y: num | np.ndarray) -> num | np.ndarray:
    """
    Example 2d function
    :param x:
    :param y:
    :return:
    """
    return np.sqrt(np.power(x, 2) + np.power(y, 2))


def _fwd2(field: np.ndarray, row: int, col: int, dx: num, dy: num, axis: str) -> num:
    """
    Second order forward difference along the given axis
    :param field:
    :param dx:
    :param dy:
    :param row:
    :param col:
    :param axis:
    :return:
    """
    if axis not in ['x', 'y']:
        raise ValueError('Invalid axis')
    if axis == 'x':
        return (field[row, col + 1] - field[row, col]) / dx
    return (field[row + 1, col] - field[row, col]) / dy


def _cnt2(field: np.ndarray, row: int, col: int, dx: num, dy: num, axis: str) -> num:
    """
    Second order center difference along the given axis
    :param field:
    :param row:
    :param col:
    :param dx:
    :param dy:
    :param axis:
    :return:
    """
    if axis not in ['x', 'y']:
        raise ValueError('Invalid axis')
    if axis == 'x':
        return (field[row, col - 1] - 2 * field[row, col] + field[row, col + 1]) \
            / (dx * dx)
    return (field[row - 1, col] - 2 * field[row, col] + field[row + 1, col]) / (dy * dy)


def _bwd2(field: np.ndarray, row: int, col: int, dx: num, dy: num, axis: str) -> num:
    """
    Second order backward difference along the given axis
    :param field:
    :param dx:
    :param dy:
    :param row:
    :param col:
    :param axis:
    :return:
    """
    if axis not in ['x', 'y']:
        raise ValueError('Invalid axis')
    if axis == 'x':
        return (field[row, col] - field[row, col - 1]) / dx
    return (field[row, col] - field[row - 1, col]) / dy



def laplace(field: np.ndarray, dx: num, dy: num) -> np.ndarray:
    """
    Laplacian of a 2d scalar field
    :param field:
    :param dx:
    :param dy:
    :return:
    """
    rows, cols = field.shape
    laplacian = np.zeros(shape=(rows, cols))
    laplacian[:, :] = field[:, :]
    for j in range(1, rows - 1):
        for i in range(1, cols - 1):
            dfdx2 = _cnt2(field, j, i, dx, dy, axis='x')
            dfdy2 = _cnt2(field, j, i, dx, dy, axis='y')
            laplacian[j, i] = dfdx2 + dfdy2

    return laplacian


def main():
    x = y = np.linspace(-5, 5, 21)
    dx = dy = x[1] - x[0]
    fun = sqroot
    xx, yy = np.meshgrid(x, y)
    field = fun(xx, yy)
    lapl = laplace(field, dx, dy)
    divgrad = div(grad(field, dx, dy), dx, dy)
    plt.contourf(x, y, lapl)
    plt.colorbar()
    _ = plt.figure()
    plt.contourf(x, y, divgrad)
    plt.colorbar()
    plt.show()

    # Assert that div(grad(field)) is the same as the Laplacian
    # print(lapl)
    # print()
    # print(divg)
    print(divgrad == lapl)


if __name__ == '__main__':
    main()
