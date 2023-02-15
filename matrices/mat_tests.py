"""
Unittests for the matrix operations
"""

import unittest
import operations as ops
import exceptions as exs


class TestMatOps(unittest.TestCase):
    def test_determine_dimension(self) -> None:
        """
        Tests for the function that determines the dimensions of a matrix
        :return:
        """
        # Matrices with a incomplete rows
        with self.assertRaises(exs.DimensionError):
            ops.determine_dimensions([[1, 2], [3]])
        with self.assertRaises(exs.DimensionError):
            ops.determine_dimensions([[1, 2], []])
        with self.assertRaises(exs.DimensionError):
            ops.determine_dimensions([[1, 2], [3, 4, 5]])

        # Matrices with different amounts of rows and columns
        mat1 = [[1, 2], [3, 4]]
        self.assertTupleEqual(ops.determine_dimensions(mat1), (2, 2))
        mat2 = [[1, 2, 3], [4, 5, 6]]
        self.assertTupleEqual(ops.determine_dimensions(mat2), (2, 3))
        mat3 = [[1, 2], [3, 4], [5, 6]]
        self.assertTupleEqual(ops.determine_dimensions(mat3), (3, 2))
        mat4 = [[1], [2], [3]]
        self.assertTupleEqual(ops.determine_dimensions(mat4), (3, 1))
        mat5 = [1, 2, 3]
        with self.assertRaises(exs.DimensionError):
            ops.determine_dimensions(mat5)
        mat6 = [[1, 2, 3]]
        self.assertTupleEqual(ops.determine_dimensions(mat6), (1, 3))
        with self.assertRaises(exs.DimensionError):
            ops.determine_dimensions([])
        with self.assertRaises(exs.DimensionError):
            ops.determine_dimensions([[]])

    def test_scalar_mult(self) -> None:
        """
        Tests for the scalar multiplication of matrices. Test for a non-scalar
        multiplier is left out since the correct type for it is type hinted.
        :return:
        """
        mul = 5
        # Multiplication of a matrix with just a single row
        mat1 = [1, 2, 3, 4]
        with self.assertRaises(exs.DimensionError):
            ops.scalar_mult(mat1, mul)
        mat2 = [[1, 2, 3, 4]]
        self.assertListEqual(ops.scalar_mult(mat2, mul), [[5, 10, 15, 20]])

        # Multiplication of a matrix with a single column
        mat2 = [[1], [2], [3], [4]]
        self.assertListEqual(ops.scalar_mult(mat2, mul), [[5], [10], [15], [20]])

        # Multiplication of empty matrices
        with self.assertRaises(exs.DimensionError):
            ops.scalar_mult([], mul)
        with self.assertRaises(exs.DimensionError):
            ops.scalar_mult([[]], mul)

        # Multiplication of a "fuller" matrix
        mat3 = [[1, 2, 3], [4, 5, 6]]
        prod = [[5, 10, 15], [20, 25, 30]]
        self.assertListEqual(ops.scalar_mult(mat3, mul), prod)



def main() -> None:
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestMatOps))
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    main()
