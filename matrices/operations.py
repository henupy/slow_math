"""
A handful of matrix operations that utilise only python's builtin
features
"""

import exceptions as exs


def scalar_mult(mat: list | list[list], mul: int | float) -> list | list[list]:
    """
    Scalar multiplication of a matrix
    :param mat: A row or a column vector
    :param mul: Multiplicator with which the matrix's elements are
    multiplied
    :return: Matrix of same shape as the param, with the elements
    multiplied by the multiplier
    """
    if type(mul) not in [int, float]:
        raise ValueError('Non-scalar multiplier')
    if not mat:
        return mat
    if type(mat[0]) == list:
        mult_mat = []
        ind = 0
        for row in mat:
            mult_mat.append([])
            for val in row:
                mult_mat[ind].append(val * mul)
            ind += 1
    else:
        mult_mat = [i * mul for i in mat]
    return mult_mat


def determine_dimensions(mat: list[list]) -> tuple:
    """
    Finds out the dimensions of a matrix
    :param mat: Matrix where each nested list is a row
    :return: tuple of (rows, columns)
    """
    rows = len(mat)
    cols = len(mat[0])
    if any(len(row) != cols for row in mat):
        raise exs.DimensionError('At least one of the rows is of different length '
                                 'than the others')
    return rows, cols


def transpose(mat: list[list]) -> list[list]:
    """
    Transposes the given matrix/vector
    :param mat:
    :return:
    """
    new_mat = []
    r, c = determine_dimensions(mat)
    if r == 1:
        for num in mat[0]:
            new_mat.append([num])
        return new_mat
    for i in range(c):
        new_mat.append([])
        for j in range(r):
            new_mat[i].append(mat[j][i])

    return new_mat


def mat_sum(mat1: list[list], mat2: list[list]) -> list[list]:
    """
    Returns the sum matrix of two matrices
    :param mat1: Matrix where the nested lists are columns
    :param mat2: Matrix where the nested lists are columns
    :return: Sum matrix, same shape as the inputs
    """
    dim1 = determine_dimensions(mat1)
    dim2 = determine_dimensions(mat2)
    assert dim1 == dim2, 'Matrices must have same shape'
    sum_mat = [[0] * dim1[1] for _ in range(dim1[0])]
    for i in range(len(mat1)):
        for j in range(len(mat1[i])):
            sum_mat[i][j] = mat1[i][j] + mat2[i][j]

    return sum_mat


def flatten(lst: list[list]) -> list:
    """
    Flattens a nested list
    :param lst: A nested list
    :return: A flattened list that has the elements of the nested
    list
    """
    flat = []
    for sublst in lst:
        flat.extend(sublst)
    return flat


def dot_prod(v1: list[list], v2: list[list]) -> float:
    """
    Function to handle vector dot production
    :param v1: Vector, where the nested lists are the rows
    :param v2: Vector, where the nested lists are the rows
    :return: Dot product of the two vectors
    """
    if isinstance(v1[0], list):
        v1 = flatten(v1)
    if isinstance(v2[0], list):
        v2 = flatten(v2)
    assert len(v1) == len(v2), 'Vectors must have same length'
    summa = 0
    for i, j in zip(v1, v2):
        summa += i * j
    return summa


def mat_mul(mat1: list[list], mat2: list[list]) -> list[list]:
    """
    Matrix multiplication
    :param mat1:
    :param mat2:
    :return:
    """
    dim1 = determine_dimensions(mat1)
    dim2 = determine_dimensions(mat2)
    assert dim1[1] == dim2[0], 'Dimension mismatch'
    prod = []
    if dim2[1] == 1:
        for i in range(dim1[1]):
            prod.append([dot_prod(mat1[i], mat2)])
        return prod
    mat2 = transpose(mat2)
    for i in range(dim2[1]):
        prod.append([])
        for j in range(dim1[1]):
            prod[i].append(dot_prod(mat1[j], mat2[i]))

    return transpose(prod)
