"""
Divergence of a 2D vector field

Some examples of vector fields are from Geeks for Geeks:
https://www.geeksforgeeks.org/how-to-plot-a-simple-vector-field-in-matplotlib/
"""

import numpy as np
import matplotlib.pyplot as plt


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


def x2y2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    :param x:
    :param y:
    :return:
    """
    field = np.zeros(shape=(x.shape[0], x.shape[1], 2))
    u = x * x
    v = y * y
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


def _fwd_div(field: np.ndarray, row: int, col: int, dx: int | float,
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
        return (field[row, col + 1, 0] - field[row, col, 0]) / dx
    return (field[row + 1, col, 1] - field[row, col, 1]) / dy


def _cnt_div(field: np.ndarray, row: int, col: int, dx: int | float,
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
        return (field[row, col + 1, 0] - field[row, col - 1, 0]) / (2 * dx)
    return (field[row + 1, col, 1] - field[row - 1, col, 1]) / (2 * dy)


def _bwd_div(field: np.ndarray, row: int, col: int, dx: int | float,
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
        return (field[row, col, 0] - field[row, col - 1, 0]) / dx
    return (field[row, col, 1] - field[row - 1, col, 1]) / dy


def div(field: np.ndarray, dx: int | float, dy: int | float) -> np.ndarray:
    """
    Discrete divergence of a 2d vector field
    :param field:
    :param dx:
    :param dy:
    :return:
    """
    rows, cols = field.shape[0], field.shape[1]
    diver = np.zeros(shape=(rows, cols))
    for j in range(rows):
        for i in range(cols):
            # Bottom left corner: use forward difference for both
            if j == 0 and i == 0:
                dfxdx = _fwd_div(field, j, i, dx, dy, axis="x")
                dfydy = _fwd_div(field, j, i, dx, dy, axis="y")
                diver[j, i] = dfxdx + dfydy

            # Bottom side (no corners): Center diff. for x, forward diff. for y
            elif j == 0 and 0 < i < (cols - 1):
                dfxdx = _cnt_div(field, j, i, dx, dy, axis="x")
                dfydy = _fwd_div(field, j, i, dx, dy, axis="y")
                diver[j, i] = dfxdx + dfydy

            # Bottom right corner: Backward diff. for x, forward diff. for y
            elif j == 0 and i == (cols - 1):
                dfxdx = _bwd_div(field, j, i, dx, dy, axis="x")
                dfydy = _fwd_div(field, j, i, dx, dy, axis="y")
                diver[j, i] = dfxdx + dfydy

            # Right side (no corners): Backward diff. for x, center diff. for y
            elif 0 < j < (rows - 1) and i == (cols - 1):
                dfxdx = _bwd_div(field, j, i, dx, dy, axis="x")
                dfydy = _cnt_div(field, j, i, dx, dy, axis="y")
                diver[j, i] = dfxdx + dfydy

            # Top right corner: Backward. diff for both
            elif j == (rows - 1) and i == (cols - 1):
                dfxdx = _bwd_div(field, j, i, dx, dy, axis="x")
                dfydy = _bwd_div(field, j, i, dx, dy, axis="y")
                diver[j, i] = dfxdx + dfydy

            # Top side (no corners): Center diff. for x, backward diff. for y
            elif j == (rows - 1) and 0 < i < (cols - 1):
                dfxdx = _cnt_div(field, j, i, dx, dy, axis="x")
                dfydy = _bwd_div(field, j, i, dx, dy, axis="y")
                diver[j, i] = dfxdx + dfydy

            # Top left corner: Forward diff. for x, backward diff. for y
            elif j == (rows - 1) and i == 0:
                dfxdx = _fwd_div(field, j, i, dx, dy, axis="x")
                dfydy = _bwd_div(field, j, i, dx, dy, axis="y")
                diver[j, i] = dfxdx + dfydy

            # Left side (no corners): Forward diff. for x, center diff. for y
            elif (rows - 1) > j > 0 == i:
                dfxdx = _fwd_div(field, j, i, dx, dy, axis="x")
                dfydy = _cnt_div(field, j, i, dx, dy, axis="y")
                diver[j, i] = dfxdx + dfydy

            # Middle cells: Center diff. for both
            else:
                dfxdx = _cnt_div(field, j, i, dx, dy, axis="x")
                dfydy = _cnt_div(field, j, i, dx, dy, axis="y")
                diver[j, i] = dfxdx + dfydy

    return diver


def visualise(x: np.ndarray, y: np.ndarray, field: np.ndarray,
              diver: np.ndarray, skip: int = 1) -> None:
    """
    Shows the vector field and the divergence in the same plot
    :param x: The x-coordinates
    :param y: The y-coordinates
    :param field: The scalar field
    :param diver: The gradient of the scalar field
    :param skip: How many data points to skip so that the gradient vectors are
        not plotted at every point. For example, if skip == 2, the vectors are
        plotted at every other data point.
    :return:
    """
    plt.contourf(x, y, diver)
    plt.colorbar()
    u, v = field[::skip, ::skip, 0], field[::skip, ::skip, 1]
    plt.streamplot(x[::skip], y[::skip], u, v, color="black")
    plt.show()


def main():
    # Create a vector field
    a, b, n = -5, 5, 101
    x = y = np.linspace(a, b, n)
    dx = dy = x[1] - x[0]
    fun = efield
    xx, yy = np.meshgrid(x, y)
    vfield = fun(xx, yy)
    diver = div(vfield, dx, dy)
    visualise(x, y, vfield, diver, 1)


if __name__ == "__main__":
    main()
