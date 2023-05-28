"""
File for custom exception(s)
"""


class DimensionError(Exception):
    """
    Exception for a situation where the matrix dimensions
    are incorrect
    """


class EmptyMatrixError(Exception):
    """
    Exception for the situation that no data is provided to form
    the matrix
    """


class InvalidRowError(Exception):
    """
    Exception for the situation that at least one of the rows in the
    matrix is of different length than the others
    """


class InvalidDataError(Exception):
    """
    Exception for the case where non-numerical data is found in the
    data used to construct the matrix
    """


class AdditionError(Exception):
    """
    Exception for the case that an invalid data type is given
    in the scalar or elementwise summation of a matrix
    """


class SubstractionError(Exception):
    """
    Exception for the case that an invalid data type is given
    in the scalar or elementwise substraction of a matrix
    """


class MultiplicationError(Exception):
    """
    Exception for the case that an invalid data type is given
    in the scalar or elementwise multiplication of a matrix
    """


class ReshapeError(Exception):
    """
    Exception for the case that the matrix can't be reshaped to the
    given shape
    """
