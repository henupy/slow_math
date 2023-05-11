"""
Unittests for the matrix operations
"""

import math
import unittest

import exceptions as exs
import operations as ops


# To type hint
from operations import matrix


def _almost_equal(mat1: matrix, mat2: matrix, eps: float = 1e-6) -> bool:
    """
    Function to check whether two arrays are equal, within a given
    margin defined by eps. The margin accounts for floating point or
    rounding errors. We can assume here that the matrices have same shape.
    :param mat1:
    :param mat2:
    :param eps:
    :return:
    """
    for row1, row2 in zip(mat1, mat2):
        for v1, v2 in zip(row1, row2):
            diff = abs(v1 - v2)
            if diff > eps:
                return False

    return True


class TestMatOps(unittest.TestCase):
    def test_validate_matrix(self) -> None:
        """
        :return:
        """
        with self.assertRaises(exs.DimensionError):
            ops.determine_dimensions([1, 2, 3])
        with self.assertRaises(exs.DimensionError):
            ops.determine_dimensions([])
        with self.assertRaises(exs.DimensionError):
            ops.determine_dimensions([[]])

    def test_determine_dimensions(self) -> None:
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
        mat6 = [[1, 2, 3]]
        self.assertTupleEqual(ops.determine_dimensions(mat6), (1, 3))

    def test_scalar_mult(self) -> None:
        """
        Tests for the scalar multiplication of matrices
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

        # Multiplication of a "fuller" matrix
        mat3 = [[1, 2, 3], [4, 5, 6]]
        prod = [[5, 10, 15], [20, 25, 30]]
        self.assertListEqual(ops.scalar_mult(mat3, mul), prod)

    def test_transpose(self) -> None:
        """
        Tests for the transpose function
        :return:
        """
        mat1 = [[1, 2], [3, 4]]
        mat1_t = [[1, 3], [2, 4]]
        self.assertListEqual(ops.transpose(mat1), mat1_t)
        self.assertListEqual(ops.transpose(mat1_t), mat1)
        mat2 = [[1, 2, 3]]
        mat2_t = [[1], [2], [3]]
        self.assertListEqual(ops.transpose(mat2), mat2_t)
        self.assertListEqual(ops.transpose(mat2_t), mat2)

    def test_mat_sum(self) -> None:
        """
        Tests for the matrix summation
        :return:
        """
        mat1 = [[1, 2, 3]]
        mat2 = [[4, 5, 6]]
        sum1 = [[5, 7, 9]]
        self.assertListEqual(ops.mat_sum(mat1, mat2), sum1)
        self.assertListEqual(ops.mat_sum(mat2, mat1), sum1)

        mat1_t = [[1], [2], [3]]
        mat2_t = [[4], [5], [6]]
        sum1_t = [[5], [7], [9]]
        self.assertListEqual(ops.mat_sum(mat1_t, mat2_t), sum1_t)
        self.assertListEqual(ops.mat_sum(mat2_t, mat1_t), sum1_t)

        mat3 = [[1, 2], [3, 4]]
        mat4 = [[5, 6], [7, 8]]
        sum2 = [[6, 8], [10, 12]]
        self.assertListEqual(ops.mat_sum(mat3, mat4), sum2)
        self.assertListEqual(ops.mat_sum(mat4, mat3), sum2)

        mat5 = [[1, 2, 3], [4, 5, 6]]
        mat6 = [[7, 8, 9], [10, 11, 12]]
        sum3 = [[8, 10, 12], [14, 16, 18]]
        self.assertListEqual(ops.mat_sum(mat5, mat6), sum3)
        self.assertListEqual(ops.mat_sum(mat6, mat5), sum3)

        mat7 = [[1, 2], [3, 4], [5, 6]]
        mat8 = [[7, 8], [9, 10], [11, 12]]
        sum4 = [[8, 10], [12, 14], [16, 18]]
        self.assertListEqual(ops.mat_sum(mat7, mat8), sum4)
        self.assertListEqual(ops.mat_sum(mat8, mat7), sum4)

        mat9 = [[1, 2], [3, 4]]
        mat10 = [[5, 6, 7], [8, 9, 10]]
        with self.assertRaises(exs.DimensionError):
            ops.mat_sum(mat9, mat10)

    def test_dot_product(self) -> None:
        """
        Tests for the dot product
        :return:
        """
        # Error cases (other than empty vectors)
        v1 = [[1, 2], [3, 4]]
        v2 = [[1, 2, 3]]
        with self.assertRaises(exs.DimensionError):
            ops.dot_prod(v1, v2)
        with self.assertRaises(exs.DimensionError):
            ops.dot_prod(v2, v1)
        v3 = [[4, 5, 6, 7]]
        with self.assertRaises(exs.DimensionError):
            ops.dot_prod(v2, v3)
        v4 = [[1], [2], [3]]
        with self.assertRaises(exs.DimensionError):
            ops.dot_prod(v3, v4)

        # Valid cases
        v5 = [[1, 2, 3]]
        v6 = [[4, 5, 6]]
        prod1 = 32
        self.assertEqual(ops.dot_prod(v5, v6), prod1)
        self.assertEqual(ops.dot_prod(v6, v5), prod1)
        v6_t = [[4], [5], [6]]
        self.assertEqual(ops.dot_prod(v5, v6_t), prod1)

    def test_mat_mul(self) -> None:
        """
        Tests for the matrix multiplication
        :return:
        """
        # Error cases
        mat1 = [[1, 2, 3], [4, 5, 6]]
        mat2 = [[1, 2, 3]]
        with self.assertRaises(exs.DimensionError):
            ops.mat_mul(mat1, mat2)
        with self.assertRaises(exs.DimensionError):
            ops.mat_mul(mat2, mat1)
        mat3 = [[1], [2], [3]]
        with self.assertRaises(exs.DimensionError):
            ops.mat_mul(mat3, mat1)

        # Valid cases
        prod1 = [[14], [32]]
        self.assertListEqual(ops.mat_mul(mat1, mat3), prod1)
        mat4 = [[1, 2, 3], [4, 5, 6]]
        mat5 = [[1, 2], [3, 4], [5, 6]]
        prod2 = [[22, 28], [49, 64]]
        prod3 = [[9, 12, 15], [19, 26, 33], [29, 40, 51]]
        self.assertListEqual(ops.mat_mul(mat4, mat5), prod2)
        self.assertListEqual(ops.mat_mul(mat5, mat4), prod3)

    def test_identity(self) -> None:
        """
        Tests for the generation of identity matrices
        :return:
        """
        # Error cases
        with self.assertRaises(ValueError):
            ops.identity_mat(-1)
        with self.assertRaises(ValueError):
            ops.identity_mat(0)

        # Valid cases
        res1 = [[1]]
        self.assertListEqual(ops.identity_mat(1), res1)
        res2 = [[1, 0], [0, 1]]
        self.assertListEqual(ops.identity_mat(2), res2)
        res3 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        self.assertListEqual(ops.identity_mat(4), res3)

    def test_mat_pow(self) -> None:
        """
        Tests for matrix power, i.e., raising a matrix to some integer
        power
        :return:
        """
        # Error cases due to invalid matrix
        with self.assertRaises(exs.DimensionError):
            ops.mat_pow(mat=[], n=1)
        with self.assertRaises(exs.DimensionError):
            ops.mat_pow(mat=[[]], n=1)
        with self.assertRaises(exs.DimensionError):
            ops.mat_pow(mat=[1, 2], n=1)
        with self.assertRaises(exs.DimensionError):
            ops.mat_pow(mat=[[1, 2], [3, 4, 5]], n=1)

        # Error case due to invalid power
        with self.assertRaises(ValueError):
            ops.mat_pow(mat=[[1, 2], [3, 4]], n=-1)

        # Valid cases
        res1 = [[1]]
        self.assertListEqual(ops.mat_pow(mat=[[5]], n=0), res1)
        self.assertListEqual(ops.mat_pow(mat=[[5]], n=1), [[5]])
        self.assertListEqual(ops.mat_pow(mat=[[5]], n=2), [[25]])

        mat1 = [[1, 2], [3, 4]]
        mul1 = ops.mat_mul(mat1=mat1, mat2=mat1)
        self.assertListEqual(ops.mat_pow(mat=mat1, n=0), [[1, 0], [0, 1]])
        self.assertListEqual(ops.mat_pow(mat=mat1, n=1), mat1)
        self.assertListEqual(ops.mat_pow(mat=mat1, n=2), mul1)
        res1 = [[37, 54], [81, 118]]
        self.assertListEqual(ops.mat_pow(mat=mat1, n=3), res1)


    def test_mat_exp(self) -> None:
        """
        Tests for the matrix exponentiation
        :return:
        """
        # Error cases
        with self.assertRaises(exs.DimensionError):
            ops.mat_exp(mat=[])
        with self.assertRaises(exs.DimensionError):
            ops.mat_exp(mat=[[]])
        with self.assertRaises(exs.DimensionError):
            ops.mat_exp(mat=[1, 2])
        with self.assertRaises(exs.DimensionError):
            ops.mat_exp(mat=[[1, 2], [3, 4, 5]])

        # Valid cases
        epsilon = 1e-6  # Tolerance due to floating point/rounding error
        m1 = [[0, -math.pi], [math.pi, 0]]
        res1 = ops.mat_exp(mat=m1)
        expected1 = [[-1, 0], [0, -1]]
        self.assertTrue(_almost_equal(mat1=res1, mat2=expected1, eps=epsilon))

        # Example from https://en.wikipedia.org/wiki/Matrix_exponential
        m2 = [[1, 4], [1, 1]]
        res2 = ops.mat_exp(mat=m2)
        t1 = (math.exp(4) + 1) / (2 * math.exp(1))
        t2 = (math.exp(4) - 1) / math.exp(1)
        t3 = (math.exp(4) - 1) / (4 * math.exp(1))
        expected2 = [[t1, t2], [t3, t1]]
        self.assertTrue(_almost_equal(mat1=res2, mat2=expected2, eps=epsilon))

        m3 = [[-3, 0, 0], [0, 4, 0], [0, 0, 1.73]]
        res3 = ops.mat_exp(mat=m3)
        expected3 = [[math.exp(-3), 0, 0], [0, math.exp(4), 0],
                     [0, 0, math.exp(1.73)]]
        self.assertTrue(_almost_equal(mat1=res3, mat2=expected3, eps=epsilon))

def main() -> None:
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestMatOps))
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    main()
