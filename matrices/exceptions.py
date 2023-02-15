"""
File for custom exceptions
"""

class DimensionError(Exception):
    """
    Exception for a situation where the matrix dimensions
    are incorrect
    """


class MatrixError(Exception):
    """
    Exception for a situation where the matrix is not a nested list
    """