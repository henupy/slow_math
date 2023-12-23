"""
Curl (or rotor) of a 2D vector field

Some examples of vector fields are from Khan Academy:
https://www.khanacademy.org/math/multivariable-calculus/
greens-theorem-and-stokes-theorem/formal-definitions-of-divergence-and-curl
/a/defining-curl

Examples are also from wikipedia: https://en.wikipedia.org/wiki/Curl_(mathematics)
"""

import numpy as np
import matplotlib.pyplot as plt


def khan(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Vector field from Khan Academy
    :param x:
    :param y:
    :return:
    """
    field = np.zeros(shape=(y.shape[0], x.shape[1], 2))
    field[:, :, 0] = -y
    field[:, :, 1] = x
    return field


def wiki1(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Example of a 2D vector field from Wikipedia
    :param x:
    :param y:
    :return:
    """
    field = np.zeros(shape=(y.shape[0], x.shape[1], 2))
    field[:, :, 0] = y
    field[:, :, 1] = -x
    return field


def wiki2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Another example of a 2d vector field from Wikipedia
    :param x:
    :param y:
    :return:
    """
    field = np.zeros(shape=(y.shape[0], x.shape[1], 2))
    field[:, :, 1] = -x * x
    return field


def gfg_field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Example for a vector field from Geeks for Geeks
    :param x:
    :param y:
    :return:
    """
    field = np.zeros(shape=(x.shape[0], x.shape[1], 2))
    xy2 = x * x + y * y
    u = -y / np.sqrt(xy2)
    v = x / xy2
    field[:, :, 0] = u[:, :]
    field[:, :, 1] = v[:, :]
    return field


def efield(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Electric field example from Geeks for Geeks
    :param x:
    :param y:
    :return:
    """
    field = np.zeros(shape=(x.shape[0], x.shape[1], 2))
    xm2 = (x - 1) * (x - 1)
    xp2 = (x + 1) * (x + 1)
    y2 = y * y
    ex = (x + 1) / (xp2 + y2) - (x - 1) / (xm2 + y2)
    ey = y / (xp2 + y2) - y / (xm2 + y2)
    field[:, :, 0] = ex[:, :]
    field[:, :, 1] = ey[:, :]
    return field


def _fwd_curl(field: np.ndarray, row: int, col: int, dx: int | float,
              dy: int | float, axis: str) -> int | float:
    """
    Forward difference for a vector field along the given axis
    :param field:
    :param dx:
    :param dy:
    :param row:
    :param col:
    :param axis:
    :return:
    """
    if axis not in ["x", "y"]:
        raise ValueError("Invalid axis")
    if axis == "x":
        return (field[row + 1, col, 0] - field[row, col, 0]) / dy
    return (field[row, col + 1, 1] - field[row, col, 1]) / dx


def _cnt_curl(field: np.ndarray, row: int, col: int, dx: int | float,
              dy: int | float, axis: str) -> int | float:
    """
    Center difference for a vector field along the given axis
    :param field:
    :param row:
    :param col:
    :param dx:
    :param dy:
    :param axis:
    :return:
    """
    if axis not in ["x", "y"]:
        raise ValueError("Invalid axis")
    if axis == "x":
        return (field[row + 1, col, 0] - field[row - 1, col, 0]) / (2 * dy)
    return (field[row, col + 1, 1] - field[row, col - 1, 1]) / (2 * dx)


def _bwd_curl(field: np.ndarray, row: int, col: int, dx: int | float,
              dy: int | float, axis: str) -> int | float:
    """
    Backward difference for a vector field along the given axis
    :param field:
    :param dx:
    :param dy:
    :param row:
    :param col:
    :param axis:
    :return:
    """
    if axis not in ["x", "y"]:
        raise ValueError("Invalid axis")
    if axis == "x":
        return (field[row, col, 0] - field[row - 1, col, 0]) / dy
    return (field[row, col, 1] - field[row, col - 1, 1]) / dx


def curl(field: np.ndarray, dx: int | float, dy: int | float) -> np.ndarray:
    """
    Discrete curl of a 2d vector field
    :param field:
    :param dx:
    :param dy:
    :return:
    """
    rows, cols = field.shape[0], field.shape[1]
    rotor = np.zeros(shape=(rows, cols))
    for j in range(rows):
        for i in range(cols):
            # Bottom left corner: use forward difference for both
            if j == 0 and i == 0:
                dfxdy = _fwd_curl(field, j, i, dx, dy, axis="x")
                dfydx = _fwd_curl(field, j, i, dx, dy, axis="y")
                rotor[j, i] = dfydx - dfxdy

            # Bottom side (no corners): Center diff. for x, forward diff. for y
            elif j == 0 and 0 < i < (cols - 1):
                dfxdy = _fwd_curl(field, j, i, dx, dy, axis="x")
                dfydx = _cnt_curl(field, j, i, dx, dy, axis="y")
                rotor[j, i] = dfydx - dfxdy

            # Bottom right corner: Backward diff. for x, forward diff. for y
            elif j == 0 and i == (cols - 1):
                dfxdy = _fwd_curl(field, j, i, dx, dy, axis="x")
                dfydx = _bwd_curl(field, j, i, dx, dy, axis="y")
                rotor[j, i] = dfydx - dfxdy

            # Right side (no corners): Backward diff. for x, center diff. for y
            elif 0 < j < (rows - 1) and i == (cols - 1):
                dfxdy = _cnt_curl(field, j, i, dx, dy, axis="x")
                dfydx = _bwd_curl(field, j, i, dx, dy, axis="y")
                rotor[j, i] = dfydx - dfxdy

            # Top right corner: Backward. diff for both
            elif j == (rows - 1) and i == (cols - 1):
                dfxdy = _bwd_curl(field, j, i, dx, dy, axis="x")
                dfydx = _bwd_curl(field, j, i, dx, dy, axis="y")
                rotor[j, i] = dfydx - dfxdy

            # Top side (no corners): Center diff. for x, backward diff. for y
            elif j == (rows - 1) and 0 < i < (cols - 1):
                dfxdy = _bwd_curl(field, j, i, dx, dy, axis="x")
                dfydx = _cnt_curl(field, j, i, dx, dy, axis="y")
                rotor[j, i] = dfydx - dfxdy

            # Top left corner: Forward diff. for x, backward diff. for y
            elif j == (rows - 1) and i == 0:
                dfxdy = _bwd_curl(field, j, i, dx, dy, axis="x")
                dfydx = _fwd_curl(field, j, i, dx, dy, axis="y")
                rotor[j, i] = dfydx - dfxdy

            # Left side (no corners): Forward diff. for x, center diff. for y
            elif (rows - 1) > j > 0 == i:
                dfxdy = _cnt_curl(field, j, i, dx, dy, axis="x")
                dfydx = _fwd_curl(field, j, i, dx, dy, axis="y")
                rotor[j, i] = dfydx - dfxdy

            # Middle cells: Center diff. for both
            else:
                dfxdy = _cnt_curl(field, j, i, dx, dy, axis="x")
                dfydx = _cnt_curl(field, j, i, dx, dy, axis="y")
                rotor[j, i] = dfydx - dfxdy

    return rotor


def visualise(x: np.ndarray, y: np.ndarray, field: np.ndarray,
              rotor: np.ndarray, skip: int = 4) -> None:
    """
    :param x:
    :param y:
    :param field:
    :param rotor:
    :param skip:
    :return:
    """
    plt.contourf(x, y, rotor)
    plt.colorbar()
    u, v = field[::skip, ::skip, 0], field[::skip, ::skip, 1]
    plt.streamplot(x[::skip], y[::skip], u, v, density=1.4, color="black")
    plt.show()


def main():
    a, b, n = -5, 5, 2001
    fun = efield
    x = y = np.linspace(a, b, n)
    dx = dy = x[1] - x[0]
    xx, yy = np.meshgrid(x, y)
    field = fun(xx, yy)
    rotor = curl(field, dx, dy)
    visualise(x, y, field, rotor)


if __name__ == "__main__":
    main()
