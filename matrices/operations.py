"""
A handful of matrix operations that utilise only python's builtin
features
"""

import exceptions as exs

# Type definition for a matrix
matrix = list[list[int | float]]


def _validate_matrix(mat: matrix) -> None:
    """
    If the matrix is incorrectly defined, raises an error. Otherwise,
    does nothing.
    :param mat:
    :return:
    """
    if not mat or not mat[0]:
        raise exs.DimensionError('Empty matrix')
    if not isinstance(mat[0], list):
        raise exs.DimensionError('Matrix must be defined as a nested list'
                                 '(even if the matrix contains just one row)')


def determine_dimensions(mat: matrix) -> tuple:
    """
    Finds out the dimensions of a matrix
    :param mat: Matrix where each nested list is a row
    :return: tuple of (rows, columns)
    """
    _validate_matrix(mat=mat)
    rows = len(mat)
    cols = len(mat[0])
    if any(len(row) != cols for row in mat):
        raise exs.DimensionError('At least one of the rows is of different length '
                                 'than the others')
    return rows, cols


def scalar_mult(mat: matrix, mul: int | float) -> matrix:
    """
    Scalar multiplication of a matrix
    :param mat: A row or a column vector
    :param mul: Multiplicator with which the matrix's elements are
    multiplied
    :return: Matrix of same shape as the param, with the elements
    multiplied by the multiplier
    """
    _validate_matrix(mat=mat)
    if type(mul) not in [int, float]:
        raise ValueError('Non-scalar multiplier')
    mult_mat = []
    ind = 0
    for row in mat:
        mult_mat.append([])
        for val in row:
            mult_mat[ind].append(val * mul)
        ind += 1
    return mult_mat


def transpose(mat: matrix) -> matrix:
    """
    Transposes the given matrix/vector
    :param mat:
    :return:
    """
    _validate_matrix(mat=mat)
    r, c = determine_dimensions(mat=mat)
    new_mat = []
    if r == 1:
        for num in mat[0]:
            new_mat.append([num])
        return new_mat
    for i in range(c):
        new_mat.append([])
        for j in range(r):
            new_mat[i].append(mat[j][i])

    return new_mat


def mat_sum(mat1: matrix, mat2: matrix) -> matrix:
    """
    Returns the sum matrix of two matrices
    :param mat1: Matrix where the nested lists are columns
    :param mat2: Matrix where the nested lists are columns
    :return: Sum matrix, same shape as the inputs
    """
    _validate_matrix(mat=mat1)
    _validate_matrix(mat=mat2)
    dim1 = determine_dimensions(mat=mat1)
    dim2 = determine_dimensions(mat=mat2)
    if dim1 != dim2:
        raise exs.DimensionError('Matrices must have same shape')
    sum_mat = [[0] * dim1[1] for _ in range(dim1[0])]
    for i in range(len(mat1)):
        for j in range(len(mat1[i])):
            sum_mat[i][j] = mat1[i][j] + mat2[i][j]

    return sum_mat


def _flatten(lst: matrix) -> list:
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


def dot_prod(v1: matrix, v2: matrix) -> float:
    """
    Function to handle vector dot production
    :param v1: Vector, where the nested lists are the rows
    :param v2: Vector, where the nested lists are the rows
    :return: Dot product of the two vectors
    """
    _validate_matrix(mat=v1)
    _validate_matrix(mat=v2)
    dim1 = determine_dimensions(mat=v1)
    dim2 = determine_dimensions(mat=v2)
    if 1 not in dim1:
        raise exs.DimensionError('v1 must be either a column or a row vector')
    if 1 not in dim2:
        raise exs.DimensionError('v2 must be either a column or a row vector')
    v1 = _flatten(lst=v1)
    v2 = _flatten(lst=v2)
    if len(v1) != len(v2):
        raise exs.DimensionError('Vectors must have same length')
    summa = 0
    for i, j in zip(v1, v2):
        summa += i * j
    return summa


def mat_mul(mat1: matrix, mat2: matrix) -> matrix:
    """
    Matrix multiplication
    :param mat1:
    :param mat2:
    :return:
    """
    _validate_matrix(mat=mat1)
    _validate_matrix(mat=mat2)
    dim1 = determine_dimensions(mat=mat1)
    dim2 = determine_dimensions(mat=mat2)
    if dim1[1] != dim2[0]:
        raise exs.DimensionError('Invalid dimensions for multiplication')
    prod = []
    if dim2[1] == 1:
        for i in range(dim1[0]):
            prod.append([dot_prod([mat1[i]], mat2)])
        return prod
    mat2 = transpose(mat=mat2)
    for i in range(dim2[1]):
        prod.append([])
        for j in range(dim1[0]):
            prod[i].append(dot_prod(v1=[mat1[j]], v2=[mat2[i]]))

    return transpose(mat=prod)


def mat_exp(mat: matrix) -> matrix:
    """

    :param mat:
    :return:
    """
    _validate_matrix(mat=mat)