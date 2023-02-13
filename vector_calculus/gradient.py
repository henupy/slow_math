"""
Gradient of a 2D scalar field

Some example fields are from Wikipedia: https://en.wikipedia.org/wiki/Gradient
"""

import numpy as np
import matplotlib.pyplot as plt

# For typing
num = int | float


def sincos(x: num | np.ndarray, y: num | np.ndarray) -> num | np.ndarray:
    """
    :param x:
    :param y:
    :return:
    """
    return np.sin(x) + np.cos(y)


def cos2(x: num | np.ndarray, y: num | np.ndarray) -> num | np.ndarray:
    """
    :param x:
    :param y:
    :return:
    """
    xcos2 = np.power(np.cos(x), 2)
    ycos2 = np.power(np.cos(y), 2)
    return -np.power(xcos2 + ycos2, 2)


def sqroot(x: num | np.ndarray, y: num | np.ndarray) -> num | np.ndarray:
    """
    :param x:
    :param y:
    :return:
    """
    return np.sqrt(np.power(x, 2) + np.power(y, 2))


def exponential(x: num | np.ndarray, y: num | np.ndarray) -> num | np.ndarray:
    """
    Example from Wikipedia
    :param x:
    :param y:
    :return:
    """
    return x * np.exp(-(np.power(x, 2) + np.power(y, 2)))


def _fwd(field: np.ndarray, row: int, col: int, dx: num, dy: num, axis: str) -> num:
    """
    Forward difference along the given axis
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


def _cnt(field: np.ndarray, row: int, col: int, dx: num, dy: num, axis: str) -> num:
    """
    Center difference along the given axis
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
        return (field[row, col + 1] - field[row, col - 1]) / (2 * dx)
    return (field[row + 1, col] - field[row - 1, col]) / (2 * dy)


def _bwd(field: np.ndarray, row: int, col: int, dx: num, dy: num, axis: str) -> num:
    """
    Backward difference along the given axis
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


def grad(field: np.ndarray, dx: num, dy: num) -> np.ndarray:
    """
    Discrete gradient of a 2d scalar field
    :param field:
    :param dx:
    :param dy:
    :return:
    """
    rows, cols = field.shape
    nabla = np.zeros(shape=(rows, cols, 2))
    for j in range(rows):
        for i in range(cols):
            # Bottom left corner: use forward difference for both
            if j == 0 and i == 0:
                nabla[j, i, 0] = _fwd(field, j, i, dx, dy, axis='x')
                nabla[j, i, 1] = _fwd(field, j, i, dx, dy, axis='y')

            # Bottom side (no corners): Center diff. for x, forward diff. for y
            elif j == 0 and 0 < i < (cols - 1):
                nabla[j, i, 0] = _cnt(field, j, i, dx, dy, axis='x')
                nabla[j, i, 1] = _fwd(field, j, i, dx, dy, axis='y')

            # Bottom right corner: Backward diff. for x, forward diff. for y
            elif j == 0 and i == (cols - 1):
                nabla[j, i, 0] = _bwd(field, j, i, dx, dy, axis='x')
                nabla[j, i, 1] = _fwd(field, j, i, dx, dy, axis='y')

            # Right side (no corners): Backward diff. for x, center diff. for y
            elif 0 < j < (rows - 1) and i == (cols - 1):
                nabla[j, i, 0] = _bwd(field, j, i, dx, dy, axis='x')
                nabla[j, i, 1] = _cnt(field, j, i, dx, dy, axis='y')

            # Top right corner: Backward. diff for both
            elif j == (rows - 1) and i == (cols - 1):
                nabla[j, i, 0] = _bwd(field, j, i, dx, dy, axis='x')
                nabla[j, i, 1] = _bwd(field, j, i, dx, dy, axis='y')

            # Top side (no corners): Center diff. for x, backward diff. for y
            elif j == (rows - 1) and 0 < i < (cols - 1):
                nabla[j, i, 0] = _cnt(field, j, i, dx, dy, axis='x')
                nabla[j, i, 1] = _bwd(field, j, i, dx, dy, axis='y')

            # Top left corner: Forward diff. for x, backward diff. for y
            elif j == (rows - 1) and i == 0:
                nabla[j, i, 0] = _fwd(field, j, i, dx, dy, axis='x')
                nabla[j, i, 1] = _bwd(field, j, i, dx, dy, axis='y')

            # Left side (no corners): Forward diff. for x, center diff. for y
            elif (rows - 1) > j > 0 == i:
                nabla[j, i, 0] = _fwd(field, j, i, dx, dy, axis='x')
                nabla[j, i, 1] = _cnt(field, j, i, dx, dy, axis='y')

            # Middle cells: Center diff. for both
            else:
                nabla[j, i, 0] = _cnt(field, j, i, dx, dy, axis='x')
                nabla[j, i, 1] = _cnt(field, j, i, dx, dy, axis='y')

    return nabla


def visualise(x: np.ndarray, y: np.ndarray, field: np.ndarray,
              nabla: np.ndarray, skip: int = 4) -> None:
    """
    Shows the field and the gradient in the same plot
    :param x: The x-coordinates
    :param y: The y-coordinates
    :param field: The scalar field
    :param nabla: The gradient of the scalar field
    :param skip: How many data points to skip so that the gradient vectors are
    not plotted at every point. For example, if skip == 2, the vectors are
    plotted at every other data point.
    :return:
    """
    u, v = nabla[::skip, ::skip, 0], nabla[::skip, ::skip, 1]
    plt.contourf(x, y, field)
    plt.colorbar()
    plt.quiver(x[::skip], y[::skip], u, v)
    plt.show()


def main() -> None:
    # Create the field
    a, b, n = -5, 5, 21
    fun = sqroot
    x = y = np.linspace(a, b, n)
    dx = dy = x[1] - x[0]
    xx, yy = np.meshgrid(x, y)
    field = fun(xx, yy)
    nabla1 = grad(field, dx, dy)
    visualise(x, y, field, nabla1, 2)


if __name__ == '__main__':
    main()
