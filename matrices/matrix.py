"""
File for a Matrix object
"""

from __future__ import annotations

import math
import random
import exceptions as exs


class Matrix:
    """
    A Matrix class that should implement the main functions of a matrix.
    Works only for 2d matrices.
    """
    def __init__(self, data: list | list[list]) -> None:
        """
        :param data: Numerical data as a list or a nested list. Used
        to construct the matrix.
        """
        # Check that the data only contains numbers and is not empty
        self._validate_data(data=data)
        self.data = data
        # The matrix is in essence validated in this step as well
        self.shape = self._determine_dimensions()

    @staticmethod
    def _validate_data(data) -> None:
        """
        Raises an error if invalid data is found
        :param data:
        :return:
        """
        if not data or (isinstance(data[0], list) and not data[0]):
            msg = 'No data provided for the matrix.'
            raise exs.EmptyMatrixError(msg)

        msg = 'Data must be only numerical.'
        if not isinstance(data[0], list):
            for val in data:
                if not isinstance(val, (int, float)):
                    raise exs.InvalidDataError(msg)
            return
        # Any empty rows are not allowed
        for row in data:
            if not row:
                raise exs.InvalidRowError('One of the rows was empty')

        # Non-numerical data is not allowed
        for row in data:
            for val in row:
                if not isinstance(val, (int, float)):
                    raise exs.InvalidDataError(msg)

    def _determine_dimensions(self) -> tuple:
        """
        Finds out the dimensions of a matrix
        :return: tuple of (rows, columns)
        """
        # Check if data is of form [1, 2, 3]
        if not isinstance(self.data[0], list):
            self.data = [self.data]
            return 1, len(self.data[0])
        # Check if data is of form [[1, 2, 3]]
        flat = self._flatten(v=self.data)
        if self.data[0] == flat:
            return 1, len(self.data[0])
        rows = len(self.data)
        cols = len(self.data[0])
        if any(len(row) != cols for row in self.data):
            msg = 'At least one of the rows of the matrix is of different ' \
                  'length than the others.'
            raise exs.InvalidRowError(msg)
        return rows, cols

    @property
    def transpose(self) -> Matrix:
        """
        :return:
        """
        r, c = self.shape
        if r == 1:
            trans_val = []
            for num in self.data[0]:
                trans_val.append([num])
            return Matrix(data=trans_val)
        trans_val = [[0] * r for _ in range(c)]
        for i in range(c):
            for j in range(r):
                trans_val[i][j] = self.data[j][i]

        return Matrix(data=trans_val)

    def reshape(self, new_shape: tuple) -> Matrix:
        """
        Reshapes the current matrix into the new given shape
        :param new_shape:
        :return:
        """
        r, c = self.shape
        new_r, new_c = new_shape
        if r * c != new_r * new_c:
            msg = f'Cannot reshape the matrix to the new given shape. The matrix ' \
                  f'has {r * c} elements while the new shape has {new_r * new_c}.'
            raise exs.ReshapeError(msg)
        new = [[0] * new_c for _ in range(new_r)]
        old = self._flatten(v=self.data)
        ind = 0
        for j in range(new_r):
            for i in range(new_c):
                new[j][i] = old[ind]
                ind += 1

        return Matrix(data=new)

    def norm(self) -> float:
        """
        Calculates the value of the norm of the (flatten) matrix
        :return:
        """
        flat = self._flatten(v=self.data)
        val = sum(i * i for i in flat) ** .5
        return val

    def _scalar_sum(self, other: int | float) -> Matrix:
        """
        :param other:
        :return:
        """
        r, c = self.shape
        sum_vals = [[0] * c for _ in range(r)]
        for j, row in enumerate(self.data):
            for i, val in enumerate(row):
                sum_vals[j][i] = val + other

        return Matrix(data=sum_vals)

    def _elem_wise_sum(self, other: Matrix) -> Matrix:
        """
        :param other:
        :return:
        """
        if self.shape != other.shape:
            msg = 'Matrices must have same shape.'
            raise exs.DimensionError(msg)
        # Initialise the values for the new matrix
        sum_vals = [[0] * self.shape[1] for _ in range(self.shape[0])]
        for j, (row1, row2) in enumerate(zip(self.data, other.data)):
            for i, (v1, v2) in enumerate(zip(row1, row2)):
                sum_vals[j][i] = v1 + v2

        return Matrix(data=sum_vals)

    def _scalar_sub(self, other: int | float) -> Matrix:
        """
        :param other:
        :return:
        """
        r, c = self.shape
        sum_vals = [[0] * c for _ in range(r)]
        for j, row in enumerate(self.data):
            for i, val in enumerate(row):
                sum_vals[j][i] = val - other

        return Matrix(data=sum_vals)

    def _elem_wise_sub(self, other: Matrix) -> Matrix:
        """
        :param other:
        :return:
        """
        if self.shape != other.shape:
            msg = 'Matrices must have same shape.'
            raise exs.DimensionError(msg)
        # Initialise the values for the new matrix
        sum_vals = [[0] * self.shape[1] for _ in range(self.shape[0])]
        for j, (row1, row2) in enumerate(zip(self.data, other.data)):
            for i, (v1, v2) in enumerate(zip(row1, row2)):
                sum_vals[j][i] = v1 - v2

        return Matrix(data=sum_vals)

    def _scalar_mul(self, other: int | float) -> Matrix:
        """
        Matrix multiplication by a scalar
        :param other:
        :return:
        """
        # Initialise a new matrix (or its values)
        r, c = self.shape
        mult_vals = [[0] * c for _ in range(r)]
        for j, row in enumerate(self.data):
            for i, val in enumerate(row):
                mult_vals[j][i] = val * other

        return Matrix(data=mult_vals)

    def _elem_wise_mul(self, other: Matrix) -> Matrix:
        """
        Elementwise multiplication of two matrices
        :param other:
        :return:
        """
        if self.shape != other.shape:
            msg = 'Matrices must have same shape.'
            raise exs.DimensionError(msg)
        # Initialise the values for the new matrix
        mult_val = [[0] * self.shape[1] for _ in range(self.shape[0])]
        for j, (row1, row2) in enumerate(zip(self.data, other.data)):
            for i, (v1, v2) in enumerate(zip(row1, row2)):
                mult_val[j][i] = v1 * v2

        return Matrix(data=mult_val)

    @staticmethod
    def _flatten(v: list[list]) -> list:
        """
        Changes the column vector into a row vector
        :param v: Vector that has multiple rows
        :return:
        """
        flat = []
        for row in v:
            flat.extend(row)
        return flat

    def _dot_prod(self, v1: list | list[list], v2: list | list[list]) \
            -> int | float:
        """
        Dot product of two vectors. Here used only as a part of the
        matrix multiplication.
        :param v1: A row or a column vector
        :param v2: A row or a column vector
        :return:
        """
        # Flatten the column vector (if there is one)
        if isinstance(v1[0], list):
            v1 = self._flatten(v=v1)
        if isinstance(v2[0], list):
            v2 = self._flatten(v=v2)
        if len(v1) != len(v2):
            raise exs.DimensionError('Vectors must have same length')
        prod = 0
        for i, j in zip(v1, v2):
            prod += i * j
        return prod

    def __add__(self, other: int | float | Matrix) -> Matrix:
        """
        Elementwise addition of two matrices
        :param other: A matrix that has the same shape as the matrix
        that this matrix is added to
        :return:
        """
        if not isinstance(other, (int, float, Matrix)):
            msg = 'The summand must be a number or another Matrix.'
            raise exs.AdditionError(msg)
        if isinstance(other, (int, float)):
            return self._scalar_sum(other=other)
        return self._elem_wise_sum(other=other)

    def __radd__(self, other: int | float) -> Matrix:
        """
        :param other:
        :return:
        """
        return self.__add__(other=other)

    def __sub__(self, other: int | float | Matrix) -> Matrix:
        """
        Elementwise substraction of two matrices
        :param other: A matrix that has the same shape as the matrix
        that this matrix is substracted from
        :return:
        """
        if not isinstance(other, (int, float, Matrix)):
            msg = 'The other part of the substraction must be a number or ' \
                  'another Matrix.'
            raise exs.SubstractionError(msg)
        if isinstance(other, (int, float)):
            return self._scalar_sub(other=other)
        return self._elem_wise_sub(other=other)

    def __rsub__(self, other: int | float) -> Matrix:
        """
        :param other:
        :return:
        """
        return self.__sub__(other=other)

    def __mul__(self, other: int | float | Matrix) -> Matrix:
        """
        Scalar multiplication of one matrix or elementwise multiplication
        of two matrices.
        :param other: Either a scalar or another matrix that has the same
        dimensions as the other one.
        :return:
        """
        if not isinstance(other, (int, float, Matrix)):
            msg = 'The multiplier must be a number or another Matrix.'
            raise exs.MultiplicationError(msg)
        if isinstance(other, (int, float)):
            return self._scalar_mul(other=other)
        return self._elem_wise_mul(other=other)

    def __rtruediv__(self, other: int | float) -> Matrix:
        """
        :param other:
        :return:
        """
        res = [[0] * self.shape[1] for _ in range(self.shape[0])]
        for j, row in enumerate(self.data):
            for i, val in enumerate(row):
                res[j][i] = other / val

        return Matrix(data=res)

    def __matmul__(self, other: Matrix) -> Matrix:
        """
        The "normal" matrix multplication
        :param other:
        :return:
        """
        r1, c1 = self.shape
        r2, c2 = other.shape
        if c1 != r2:
            raise exs.DimensionError('Invalid dimensions for multiplication')
        prod = []
        if c2 == 1:
            for i in range(r1):
                prod.append([self._dot_prod(v1=self.data[i], v2=other.data)])
            return Matrix(data=prod)
        mat2 = other.transpose
        for i in range(c2):
            prod.append([])
            for j in range(r1):
                prod[i].append(self._dot_prod(v1=self.data[j], v2=mat2.data[i]))

        return Matrix(data=prod).transpose

    def __pow__(self, power: int) -> Matrix:
        """
        Raises the given matrix to the given integer power
        :param power: An integer power to which raise the matrix (must be positive)
        :return:
        """
        if self.shape[0] != self.shape[1]:
            msg = 'The matrix must be a square matrix.'
            raise exs.DimensionError(msg)
        # Let's allow only positive powers for now
        if power < 0:
            msg = 'Only positive powers (power >= 0) are allowed.'
            raise ValueError(msg)
        if not isinstance(power, int):
            power = int(power)
            print('WARNING: The given power was converted to an integer.')
        # If n == 0, the result is a identity matrix with the same dimensions
        # of the given matrix
        if power == 0:
            return identity_matrix(n=self.shape[0])
        res = Matrix(self.data)  # Create a copy of the original matrix
        for _ in range(power - 1):
            res = self @ res
        return res

    def __neg__(self) -> Matrix:
        """
        :return:
        """
        res = [[0] * self.shape[1] for _ in range(self.shape[0])]
        for j, row in enumerate(self.data):
            for i, val in enumerate(row):
                res[j][i] = -val

        return Matrix(data=res)

    def __eq__(self, other: Matrix) -> bool:
        """
        Check if two matrices are equal
        :param other:
        :return:
        """
        # Quick check using the shapes
        if self.shape != other.shape:
            return False
        for r1, r2 in zip(self.data, other.data):
            for v1, v2 in zip(r1, r2):
                if v1 != v2:
                    return False

        return True

    def __getitem__(self, indices: int) -> int | float | list:
        """
        :param indices:
        :return:
        """
        return self.data[indices]

    def __str__(self) -> str:
        """
        Some kind of a representation of the matrix
        :return:
        """
        # If we have only one row, we can use the __str__ of the list-class
        if self.shape[0] == 1:
            return list.__str__(self.data)
        # If we have multiple rows, let's try to print them on new lines
        s = '['  # Initialise a string
        for i in range(self.shape[0] - 1):
            s += list.__str__(self.data[i]) + '\n'
        # Add the last row without a new line character
        s += list.__str__(self.data[-1]) + ']'
        return s


def identity_matrix(n: int) -> Matrix:
    """
    Generates a nxn identity matrix
    :param n: Size of the matrix (must be greater than zero)
    :return:
    """
    if n <= 0:
        raise ValueError('Too small size for the matrix. n must be > 0')
    if n == 1:
        return Matrix(data=[[1]])
    mat = [[0] * n for _ in range(n)]
    for i in range(n):
        mat[i][i] = 1
    return Matrix(data=mat)


def zeros(shape: tuple) -> Matrix:
    """
    Returns a matrix of the given shape with all values set to zero
    :param shape:
    :return:
    """
    r, c = shape
    return Matrix(data=[[0] * c for _ in range(r)])


def ones(shape: tuple) -> Matrix:
    """
    Returns a matrix of the given shape with all values set to one
    :param shape:
    :return:
    """
    r, c = shape
    return Matrix(data=[[1] * c for _ in range(r)])


def rand(shape: tuple) -> Matrix:
    """
    Returns a matrix of the given shape filled with random values
    in the range [0, 1)
    :param shape:
    :return:
    """
    r, c = shape
    nums = [[random.random() for _ in range(c)] for _ in range(r)]
    return Matrix(data=nums)


def _relative_diff(mat1: Matrix, mat2: Matrix) -> Matrix:
    """
    Calculates the relative difference of two matrices elementwise
    :param mat1: Matrix of any shape
    :param mat2: Matrix of any shape
    :return: A matrix of the same shape as the matrices passed as params,
        with each element containing the relative difference of the two matrices'
        elements at the corresponding locations.
    """
    if mat1.shape != mat2.shape:
        msg = f'The matrices must be of same shape. Now mat1 is of shape' \
              f'{mat1.shape} and mat2 is {mat2.shape}.'
        raise exs.DimensionError(msg)
    diff = []
    ind = 0
    for r1, r2 in zip(mat1.data, mat2.data):
        diff.append([])
        for v1, v2 in zip(r1, r2):
            if v1 != 0:
                diff[ind].append((v2 - v1) / v1)
            else:
                diff[ind].append((v2 - v1))
        ind += 1
    return Matrix(data=diff)


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


def elem_exp(mat: Matrix) -> Matrix:
    """
    e to the power of each element of the matrix
    :param mat:
    :return:
    """
    res = [[0.] * mat.shape[1] for _ in range(mat.shape[0])]
    for j, row in enumerate(mat.data):
        for i, val in enumerate(row):
            res[j][i] = math.exp(val)

    return Matrix(data=res)


def mat_exp(mat: Matrix, rtol: int | float = 1e-9,
            iter_limit: int = 1000) -> Matrix:
    """
    Matrix exponentiation, i.e., raising e to the power of the given
    matrix. The solution is computed using the matrix power series.
    The solution is deemed to be converged when the relative error is
    below the given tolerance.
    :param mat: A square matrix
    :param rtol: The tolerance for the relative error. Determines how
    accurate the result will be. Defaults to 1e-9.
    :param iter_limit: A limit for the amount of iterations to do in
    case the result would not converge for some reason. Prevents the
    iteration from running infinitely. Defaults to 1000.
    :return:
    """
    r, c = mat.shape
    if r != c:
        msg = f'Matrix must be a square matrix. Now got {r}x{c}'
        raise exs.DimensionError(msg)
    error, n = 1, 0
    # Initialise two result matrices to keep track of the iteration
    new_res = zeros(shape=mat.shape)
    old_res = zeros(shape=mat.shape)
    while error > rtol and n < iter_limit:
        # Calculate the result for the current iteration
        f = _factorial(n=n)
        exp = (mat ** n) * (1 / f)
        new_res = exp + old_res
        # Calculate the relative difference of the old and new results
        diff = _relative_diff(mat1=old_res, mat2=new_res)
        # The norm of the difference is the "total" error
        error = diff.norm()
        # Save the result of this iteration
        old_res = new_res
        n += 1
    return new_res
