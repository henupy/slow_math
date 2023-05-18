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

class InvalidMultiplierError(Exception):
    """
    Exception for the case that a non-numerical multiplier is given
    in the scalar multiplication of a matrix
    """
