"""
A handful of matrix operations that utilise only python's builtin
features
"""

def scalar_mult(v: list | list[list], m: int | float) -> list | list[list]:
    """
    Function to multiply a vector with a scalar
    :param v: A row or a column vector
    :param m: Multiplicator with which the vector's elements are
    multiplied
    :return: Vector of same shape as the param, with the elements
    multiplied by the multiplier
    """
    if type(v[0]) == list:
        mult_vect = []
        ind = 0
        for vect in v:
            mult_vect.append([])
            for val in vect:
                mult_vect[ind].append(val * m)
            ind += 1
    else:
        mult_vect = [i * m for i in v]
    return mult_vect


def vect_sum(v1: list | list[list], v2: list | list[list]) -> list | list[list]:
    """
    Sum of two vectors
    :param v1: A row or a column vector
    :param v2: A row or a column vector
    :return: Vector of same shape as the params, with the elements
    of the vectors summed together
    """
    assert len(v1) == len(v2), 'Vectors must be same length'
    sum_v = [i + j for i, j in zip(v1, v2)]
    return sum_v


def determine_dimensions(mat: list[list]) -> tuple:
    """
    Finds out the dimensions of a matrix
    :param mat: Matrix where each nested list is a row
    :return: tuple of (rows, columns)
    """
    rows = len(mat)
    cols = len(mat[0])
    assert all(len(row) == cols for row in mat), 'Some row is longer ' \
                                                 'than another'
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
