"""
A handful of matrix operations that utilise only python's builtin
features
"""

import exceptions as exs

from copy import deepcopy

# Type definition for a valid matrix
matrix = list[list[int | float]]


def _factorial(n: int) -> int:
    """
    Nth factorial. This is needed in the matrix exponentiation.
    :param n: A positive integer (or zero)
    :return:
    """
    # Let's not handle negative numbers now
    if n < 1:
        return 1
    prod = 1
    for i in range(1, n + 1):
        prod *= i
    return prod


def _validate_matrix(mat: matrix) -> None:
    """
    Raises an error if the matrix is incorrectly defined
    :param mat: Matrix where each nested list corresponds to a row
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
    :param mat: Matrix where each nested list corresponds to a row
    :return: tuple of (rows, columns)
    """
    _validate_matrix(mat=mat)
    rows = len(mat)
    cols = len(mat[0])
    if any(len(row) != cols for row in mat):
        raise exs.DimensionError('At least one of the rows is of different'
                                 'length than the others')
    return rows, cols


def scalar_mult(mat: matrix, mul: int | float) -> matrix:
    """
    Scalar multiplication of a matrix
    :param mat: Matrix where each nested list corresponds to a row
    :param mul: Multiplicator with which the matrix's elements are
        multiplied. Must be a number.
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
    Transposes the given matrix
    :param mat: Matrix where each nested list corresponds to a row
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
    :param mat1: Matrix where each nested list corresponds to a row
    :param mat2: Matrix where each nested list corresponds to a row
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


def _flatten(mat: matrix) -> list:
    """
    Flattens the matrix
    :param mat: Matrix where each nested list corresponds to a row
    :return: A flattened matrix that has the elements of the nested
    matrix
    """
    flat = []
    for row in mat:
        flat.extend(row)
    return flat


def dot_prod(v1: matrix, v2: matrix) -> int | float:
    """
    Dot product of two vectors
    :param v1: Vector, where the nested lists are the rows
    :param v2: Vector, where the nested lists are the rows
    :return:
    """
    _validate_matrix(mat=v1)
    _validate_matrix(mat=v2)
    dim1 = determine_dimensions(mat=v1)
    dim2 = determine_dimensions(mat=v2)
    if 1 not in dim1:
        raise exs.DimensionError('v1 must be either a column or a row vector')
    if 1 not in dim2:
        raise exs.DimensionError('v2 must be either a column or a row vector')
    v1 = _flatten(mat=v1)
    v2 = _flatten(mat=v2)
    if len(v1) != len(v2):
        raise exs.DimensionError('Vectors must have same length')
    summa = 0
    for i, j in zip(v1, v2):
        summa += i * j
    return summa


def mat_mul(mat1: matrix, mat2: matrix) -> matrix:
    """
    The common matrix multiplication
    :param mat1: Matrix where each nested list corresponds to a row
    :param mat2: Matrix where each nested list corresponds to a row
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


def identity_mat(n: int) -> matrix:
    """
    Generates a nxn identity matrix
    :param n: Size of the matrix (must be greater than zero)
    :return:
    """
    if n <= 0:
        raise ValueError('Too small size for the matrix. n must be > 0')
    if n == 1:
        return [[1]]
    mat = [[0] * n for _ in range(n)]
    for i in range(n):
        mat[i][i] = 1
    return mat


def mat_pow(mat: matrix, n: int) -> matrix:
    """
    Raises the given matrix to the power of n
    :param mat: Matrix where each nested list corresponds to a row
    :param n: An integer power to which raise the matrix (must be positive)
    :return:
    """
    # Validate the matrix and check the dimensions
    _validate_matrix(mat=mat)
    dim = determine_dimensions(mat=mat)
    if dim[0] != dim[1]:
        msg = f'Matrix must be a square matrix. Now got {dim[0]}x{dim[1]}'
        raise exs.DimensionError(msg)
    # Let's allow only positive powers for now
    if n < 0:
        msg = 'Only positive powers (n >= 0) are allowed.'
        raise ValueError(msg)
    # If n == 0, the result is a identity matrix with the same dimensions
    # of the given matrix
    if n == 0:
        return identity_mat(n=dim[0])
    res = deepcopy(mat)
    for _ in range(n - 1):
        res = mat_mul(mat1=mat, mat2=res)
    return res


def _relative_diff(mat1: matrix, mat2: matrix) -> matrix:
    """
    Calculates the relative difference of two matrices elementwise
    :param mat1: Matrix where each nested list corresponds to a row
    :param mat2: Matrix where each nested list corresponds to a row
    :return: A matrix of the same shape as the matrices passed as params,
        with each element containing the relative difference of the two matrices'
        elements at the corresponding locations.
    """
    _validate_matrix(mat=mat1)
    _validate_matrix(mat=mat2)
    dim1 = determine_dimensions(mat=mat1)
    dim2 = determine_dimensions(mat=mat2)
    if dim1 != dim2:
        msg = f'The matrices must be of same shape. Now mat1 is of shape' \
              f'{dim1} and mat2 is {dim2}.'
        raise exs.DimensionError(msg)
    diff = []
    ind = 0
    for r1, r2 in zip(mat1, mat2):
        diff.append([])
        for v1, v2 in zip(r1, r2):
            if v1 != 0:
                diff[ind].append((v2 - v1) / v1)
            else:
                diff[ind].append((v2 - v1))
        ind += 1
    return diff


def _norm(mat: matrix) -> float:
    """
    Calculates the value of the norm of the (flatten) matrix
    :param mat:
    :return:
    """
    _validate_matrix(mat=mat)
    flat = _flatten(mat=mat)
    val = sum(i * i for i in flat) ** 0.5
    return val


def mat_exp(mat: matrix, rtol: int | float = 1e-9,
            iter_limit: int = 1000) -> matrix:
    """
    Matrix exponentiation, i.e., raising e to the power of the given
    matrix. The solution is computed using the matrix power series.
    The solution is deemed to be converged when the relative error is
    below the given tolerance.
    :param mat:
    :param rtol:
    :param iter_limit:
    :return:
    """
    # Check that the matrix is valid and has valid dimensions for exponentiation
    _validate_matrix(mat=mat)
    dim = determine_dimensions(mat=mat)
    if dim[0] != dim[1]:
        msg = f'Matrix must be a square matrix. Now got {dim[0]}x{dim[1]}'
        raise exs.DimensionError(msg)
    error, n = 1, 0
    # Initialise two result matrices
    new_res = [[0] * dim[0] for _ in range(dim[0])]
    old_res = deepcopy(new_res)
    while error > rtol and n < iter_limit:
        # Calculate the result for the current iteration
        f = _factorial(n=n)
        exp = scalar_mult(mat=mat_pow(mat=mat, n=n), mul=1/f)
        new_res = mat_sum(mat1=exp, mat2=old_res)
        # Calculate the relative difference of the old and new results
        diff = _relative_diff(mat1=old_res, mat2=new_res)
        # The norm of the difference is the "total" error
        error = _norm(mat=diff)
        # Save the result of this iteration
        old_res = new_res
        n += 1
    return new_res
