"""
Unittests for the matrix operations
"""

import math
import unittest
import matrix
import exceptions as exs


def _almost_equal(mat1: matrix.Matrix, mat2: matrix.Matrix,
                  eps: float = 1e-6) -> bool:
    """
    Function to check whether two arrays are equal, within a given
    margin defined by eps. The margin accounts for floating point or
    rounding errors. We can assume here that the matrices have same shape.
    :param mat1:
    :param mat2:
    :param eps:
    :return:
    """
    for row1, row2 in zip(mat1.data, mat2.data):
        for v1, v2 in zip(row1, row2):
            diff = abs(v1 - v2)
            if diff > eps:
                return False

    return True


class TestMatrix(unittest.TestCase):
    def test_matrix_creation(self) -> None:
        """
        Test that only valid matrices can be created, and the shape
        of the created matrix
        :return:
        """
        # Errors
        with self.assertRaises(exs.EmptyMatrixError):
            matrix.Matrix(data=[])
        with self.assertRaises(exs.EmptyMatrixError):
            matrix.Matrix(data=[[]])
        with self.assertRaises(exs.InvalidDataError):
            matrix.Matrix(data=['a', 'b', 'c'])
        with self.assertRaises(exs.InvalidRowError):
            matrix.Matrix(data=[[1, 2], [3]])
        with self.assertRaises(exs.InvalidRowError):
            matrix.Matrix(data=[[1, 2], []])
        with self.assertRaises(exs.InvalidRowError):
            matrix.Matrix(data=[[1, 2], [3, 4, 5]])

        # Test that the shape is correct
        m1 = matrix.Matrix(data=[[1], [2]])
        self.assertTupleEqual(m1.shape, (2, 1))
        m2 = matrix.Matrix(data=[[1, 2]])
        self.assertTupleEqual(m2.shape, (1, 2))
        m3 = matrix.Matrix(data=[1, 2])
        self.assertTupleEqual(m3.shape, (1, 2))
        m4 = matrix.Matrix(data=[[1, 2, 3], [4, 5, 6]])
        self.assertTupleEqual(m4.shape, (2, 3))

    def test_transpose(self) -> None:
        """
        Tests for the transpose function
        :return:
        """
        mat1 = matrix.Matrix(data=[[1, 2], [3, 4]])
        mat1_t = matrix.Matrix(data=[[1, 3], [2, 4]])
        self.assertTrue(mat1.transpose == mat1_t)
        self.assertTrue(mat1_t.transpose == mat1)
        mat2 = matrix.Matrix(data=[[1, 2, 3]])
        mat2_t = matrix.Matrix(data=[[1], [2], [3]])
        self.assertTrue(mat2.transpose == mat2_t)
        self.assertTrue(mat2_t.transpose == mat2)
        mat3 = matrix.Matrix(data=[1, 2, 3])
        mat3_t = matrix.Matrix(data=[[1], [2], [3]])
        self.assertTrue(mat3.transpose == mat3_t)
        self.assertTrue(mat3_t.transpose == mat3)

    def test_mat_sum(self) -> None:
        """
        Tests for the matrix summation
        :return:
        """
        mat1 = matrix.Matrix(data=[[1, 2], [3, 4]])
        mat2 = matrix.Matrix(data=[[5, 6, 7], [8, 9, 10]])
        with self.assertRaises(exs.DimensionError):
            _ = mat1 + mat2

        mat3 = matrix.Matrix(data=[[1, 2, 3]])
        mat4 = matrix.Matrix(data=[[4, 5, 6]])
        sum1 = matrix.Matrix(data=[[5, 7, 9]])
        self.assertTrue(mat3 + mat4 == sum1)
        self.assertTrue(mat4 + mat3 == sum1)

        mat3_t = matrix.Matrix(data=[[1], [2], [3]])
        mat4_t = matrix.Matrix(data=[[4], [5], [6]])
        sum1_t = matrix.Matrix(data=[[5], [7], [9]])
        self.assertTrue(mat3_t + mat4_t == sum1_t)
        self.assertTrue(mat3_t + mat4_t == sum1_t)

        mat5 = matrix.Matrix(data=[[1, 2], [3, 4]])
        mat6 = matrix.Matrix(data=[[5, 6], [7, 8]])
        sum2 = matrix.Matrix(data=[[6, 8], [10, 12]])
        self.assertTrue(mat5 + mat6 == sum2)
        self.assertTrue(mat6 + mat5 == sum2)

        mat7 = matrix.Matrix(data=[[1, 2, 3], [4, 5, 6]])
        mat8 = matrix.Matrix(data=[[7, 8, 9], [10, 11, 12]])
        sum3 = matrix.Matrix(data=[[8, 10, 12], [14, 16, 18]])
        self.assertTrue(mat7 + mat8 == sum3)
        self.assertTrue(mat8 + mat7 == sum3)

        mat9 = matrix.Matrix(data=[[1, 2], [3, 4], [5, 6]])
        mat10 = matrix.Matrix(data=[[7, 8], [9, 10], [11, 12]])
        sum4 = matrix.Matrix(data=[[8, 10], [12, 14], [16, 18]])
        self.assertTrue(mat9 + mat10 == sum4)
        self.assertTrue(mat10 + mat9 == sum4)

    def test_scalar_mult(self) -> None:
        """
        Tests for the scalar multiplication of matrices
        :return:
        """
        mul = 5
        # Multiplication of a matrix with just a single row
        mat1 = matrix.Matrix(data=[1, 2, 3, 4])
        res1 = mat1 * mul
        mat2 = matrix.Matrix(data=[[1, 2, 3, 4]])
        res2 = mat2 * mul
        self.assertListEqual(res1.data, [[5, 10, 15, 20]])
        self.assertListEqual(res1.data, res2.data)

        # Multiplication of a matrix with a single column
        mat3 = matrix.Matrix(data=[[1], [2], [3], [4]])
        res3 = mat3 * mul
        self.assertListEqual(res3.data, [[5], [10], [15], [20]])

        # Multiplication of a "fuller" matrix
        mat4 = matrix.Matrix(data=[[1, 2, 3], [4, 5, 6]])
        res4 = mat4 * mul
        prod = [[5, 10, 15], [20, 25, 30]]
        self.assertListEqual(res4.data, prod)

    def test_elementwise_mul(self) -> None:
        """
        Tests for the elementwise multiplication of two matrices
        :return:
        """
        # Test with incorrect shapes
        mat1 = matrix.Matrix(data=[[1, 2], [3, 4]])
        mat2 = matrix.Matrix(data=[[1, 2, 3], [4, 5, 6]])
        mat3 = matrix.Matrix(data=[[1], [2]])
        mat4 = matrix.Matrix(data=[[1, 2]])
        with self.assertRaises(exs.DimensionError):
            _ = mat1 * mat2
        with self.assertRaises(exs.DimensionError):
            _ = mat2 * mat1
        with self.assertRaises(exs.DimensionError):
            _ = mat3 * mat4
        with self.assertRaises(exs.DimensionError):
            _ = mat4 * mat3

        # Some valid cases
        mat5 = matrix.Matrix(data=[[2, 2], [2, 2]])
        res1 = matrix.Matrix(data=[[2, 4], [6, 8]])
        self.assertTrue(mat1 * mat5 == res1)
        self.assertTrue(mat5 * mat1 == res1)

        res2 = matrix.Matrix(data=[[1], [4]])
        self.assertTrue(mat4.transpose * mat3 == res2)
        self.assertTrue(mat3 * mat4.transpose == res2)

        mat6 = matrix.Matrix(data=[[2, 2, 2], [2, 2, 2]])
        res3 = matrix.Matrix(data=[[2, 4, 6], [8, 10, 12]])
        self.assertTrue(mat6 * mat2 == res3)
        self.assertTrue(mat2 * mat6 == res3)

    def test_mat_mul(self) -> None:
        """
        Tests for the matrix multiplication
        :return:
        """
        # Error cases
        mat1 = matrix.Matrix(data=[[1, 2, 3], [4, 5, 6]])
        mat2 = matrix.Matrix(data=[[1, 2, 3]])
        with self.assertRaises(exs.DimensionError):
            _ = mat1 @ mat2
        with self.assertRaises(exs.DimensionError):
            _ = mat2 @ mat1
        mat3 = matrix.Matrix(data=[[1], [2], [3]])
        with self.assertRaises(exs.DimensionError):
            _ = mat3 @ mat1

        # Valid cases
        prod1 = matrix.Matrix(data=[[14], [32]])
        self.assertTrue(mat1 @ mat3 == prod1)
        mat4 = matrix.Matrix(data=[[1, 2, 3], [4, 5, 6]])
        mat5 = matrix.Matrix(data=[[1, 2], [3, 4], [5, 6]])
        prod2 = matrix.Matrix(data=[[22, 28], [49, 64]])
        prod3 = matrix.Matrix(data=[[9, 12, 15], [19, 26, 33], [29, 40, 51]])
        self.assertTrue(mat4 @ mat5 == prod2)
        self.assertTrue(mat5 @ mat4 == prod3)

    def test_identity(self) -> None:
        """
        Tests for the generation of identity matrices
        :return:
        """
        # Error cases
        with self.assertRaises(ValueError):
            matrix.identity_matrix(n=-1)
        with self.assertRaises(ValueError):
            matrix.identity_matrix(n=0)

        # Valid cases
        res1 = [[1]]
        self.assertListEqual(matrix.identity_matrix(n=1).data, res1)
        res2 = [[1, 0], [0, 1]]
        self.assertListEqual(matrix.identity_matrix(n=2).data, res2)
        res3 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        self.assertListEqual(matrix.identity_matrix(n=4).data, res3)

    def test_mat_pow(self) -> None:
        """
        Tests for matrix power, i.e., raising a matrix to some integer
        power
        :return:
        """
        # Error case due to invalid power
        with self.assertRaises(ValueError):
            _ = matrix.Matrix(data=[[1, 2], [3, 4]]) ** -1

        # Error case due to non-square matrix
        with self.assertRaises(exs.DimensionError):
            _ = matrix.Matrix(data=[[1, 2, 3], [4, 5, 6]]) ** 1

        # Valid cases
        res1 = matrix.Matrix(data=[[1]])
        res2 = matrix.Matrix(data=[[5]])
        res3 = matrix.Matrix(data=[25])  # Test also without a nested row
        self.assertTrue(matrix.Matrix(data=[[5]]) ** 0 == res1)
        self.assertTrue(matrix.Matrix(data=[[5]]) ** 1 == res2)
        self.assertTrue(matrix.Matrix(data=[[5]]) ** 2 == res3)

        mat1 = matrix.Matrix(data=[[1, 2], [3, 4]])
        mul1 = mat1 @ mat1
        self.assertTrue(mat1 ** 0 == matrix.identity_matrix(n=2))
        self.assertTrue(mat1 ** 1 == mat1)
        self.assertTrue(mat1 ** 2 == mul1)
        res4 = matrix.Matrix(data=[[37, 54], [81, 118]])
        self.assertTrue(mat1 ** 3 == res4)

    def test_mat_exp(self) -> None:
        """
        Tests for the matrix exponentiation
        :return:
        """
        # Error cases

        # Valid cases
        epsilon = 1e-6  # Tolerance due to floating point/rounding error
        m1 = matrix.Matrix(data=[[0, -math.pi], [math.pi, 0]])
        res1 = matrix.mat_exp(mat=m1)
        expected1 = matrix.Matrix(data=[[-1, 0], [0, -1]])
        self.assertTrue(_almost_equal(mat1=res1, mat2=expected1, eps=epsilon))

        # Example from https://en.wikipedia.org/wiki/Matrix_exponential
        m2 = matrix.Matrix(data=[[1, 4], [1, 1]])
        res2 = matrix.mat_exp(mat=m2)
        t1 = (math.exp(4) + 1) / (2 * math.exp(1))
        t2 = (math.exp(4) - 1) / math.exp(1)
        t3 = (math.exp(4) - 1) / (4 * math.exp(1))
        expected2 = matrix.Matrix(data=[[t1, t2], [t3, t1]])
        self.assertTrue(_almost_equal(mat1=res2, mat2=expected2, eps=epsilon))

        m3 = matrix.Matrix([[-3, 0, 0], [0, 4, 0], [0, 0, 1.73]])
        res3 = matrix.mat_exp(mat=m3)
        expected3 = [[math.exp(-3), 0, 0], [0, math.exp(4), 0],
                     [0, 0, math.exp(1.73)]]
        expected3 = matrix.Matrix(data=expected3)
        self.assertTrue(_almost_equal(mat1=res3, mat2=expected3, eps=epsilon))

def main() -> None:
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestMatrix))
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    main()
