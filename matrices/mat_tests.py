"""
Unittests for the matrix operations
"""

import math
import unittest
import matrix as mx
import exceptions as exs


def _almost_equal(mat1: mx.Matrix, mat2: mx.Matrix,
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
            mx.Matrix(data=[])
        with self.assertRaises(exs.EmptyMatrixError):
            mx.Matrix(data=[[]])
        with self.assertRaises(exs.InvalidDataError):
            mx.Matrix(data=['a', 'b', 'c'])
        with self.assertRaises(exs.InvalidRowError):
            mx.Matrix(data=[[1, 2], [3]])
        with self.assertRaises(exs.InvalidRowError):
            mx.Matrix(data=[[1, 2], []])
        with self.assertRaises(exs.InvalidRowError):
            mx.Matrix(data=[[1, 2], [3, 4, 5]])

        # Test that the shape is correct
        m1 = mx.Matrix(data=[[1], [2]])
        self.assertTupleEqual(m1.shape, (2, 1))
        m2 = mx.Matrix(data=[[1, 2]])
        self.assertTupleEqual(m2.shape, (1, 2))
        m3 = mx.Matrix(data=[1, 2])
        self.assertTupleEqual(m3.shape, (1, 2))
        m4 = mx.Matrix(data=[[1, 2, 3], [4, 5, 6]])
        self.assertTupleEqual(m4.shape, (2, 3))

    def test_transpose(self) -> None:
        """
        Tests for the transpose function
        :return:
        """
        mat1 = mx.Matrix(data=[[1, 2], [3, 4]])
        mat1_t = mx.Matrix(data=[[1, 3], [2, 4]])
        self.assertTrue(mat1.transpose == mat1_t)
        self.assertTrue(mat1_t.transpose == mat1)
        mat2 = mx.Matrix(data=[[1, 2, 3]])
        mat2_t = mx.Matrix(data=[[1], [2], [3]])
        self.assertTrue(mat2.transpose == mat2_t)
        self.assertTrue(mat2_t.transpose == mat2)
        mat3 = mx.Matrix(data=[1, 2, 3])
        mat3_t = mx.Matrix(data=[[1], [2], [3]])
        self.assertTrue(mat3.transpose == mat3_t)
        self.assertTrue(mat3_t.transpose == mat3)

    def test_reshape(self) -> None:
        """
        Tests for reshaping a matrix
        :return:
        """
        # Error cases
        mat1 = mx.Matrix(data=[[1, 2]])
        with self.assertRaises(exs.ReshapeError):
            mat1.reshape(new_shape=(2, 2))
        with self.assertRaises(ValueError):
            mat1.reshape(new_shape=())
        mat2 = mx.Matrix(data=[[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(exs.ReshapeError):
            mat2.reshape(new_shape=(3, 3))

        # Valid cases
        s1 = (2, 1)
        mat1_r = mat1.reshape(new_shape=s1)
        self.assertTupleEqual(mat1_r.shape, s1)

        s2 = (3, 2)
        mat2_r = mat2.reshape(new_shape=s2)
        self.assertTupleEqual(mat2_r.shape, s2)
        mat2_r2 = mat2_r.reshape(new_shape=mat2.shape)
        self.assertTupleEqual(mat2_r2.shape, mat2.shape)
        s3 = (6, 1)
        mat2_r3 = mat2.reshape(new_shape=s3)
        self.assertTupleEqual(mat2_r3.shape, s3)

        s4 = (5, 2)
        mat3 = mx.Matrix(data=[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        mat3_r = mat3.reshape(new_shape=s4)
        self.assertTupleEqual(mat3_r.shape, s4)
        mat3_rr = mat3_r.reshape(new_shape=mat3.shape)
        self.assertTrue(mat3_rr == mat3)

    def test_mat_sum(self) -> None:
        """
        Tests for addition of two matrices and for the
        addition of a scalar number and a matrix
        :return:
        """
        mat1 = mx.Matrix(data=[[1, 2], [3, 4]])
        mat2 = mx.Matrix(data=[[5, 6, 7], [8, 9, 10]])
        with self.assertRaises(exs.DimensionError):
            _ = mat1 + mat2

        mat3 = mx.Matrix(data=[[1, 2, 3]])
        mat4 = mx.Matrix(data=[[4, 5, 6]])
        res1 = mx.Matrix(data=[[5, 7, 9]])
        self.assertTrue(mat3 + mat4 == res1)
        self.assertTrue(mat4 + mat3 == res1)

        mat3_t = mx.Matrix(data=[[1], [2], [3]])
        mat4_t = mx.Matrix(data=[[4], [5], [6]])
        res1_t = mx.Matrix(data=[[5], [7], [9]])
        self.assertTrue(mat3_t + mat4_t == res1_t)
        self.assertTrue(mat3_t + mat4_t == res1_t)

        mat5 = mx.Matrix(data=[[1, 2], [3, 4]])
        mat6 = mx.Matrix(data=[[5, 6], [7, 8]])
        res2 = mx.Matrix(data=[[6, 8], [10, 12]])
        self.assertTrue(mat5 + mat6 == res2)
        self.assertTrue(mat6 + mat5 == res2)

        mat7 = mx.Matrix(data=[[1, 2, 3], [4, 5, 6]])
        mat8 = mx.Matrix(data=[[7, 8, 9], [10, 11, 12]])
        res3 = mx.Matrix(data=[[8, 10, 12], [14, 16, 18]])
        self.assertTrue(mat7 + mat8 == res3)
        self.assertTrue(mat8 + mat7 == res3)

        mat9 = mx.Matrix(data=[[1, 2], [3, 4], [5, 6]])
        mat10 = mx.Matrix(data=[[7, 8], [9, 10], [11, 12]])
        res4 = mx.Matrix(data=[[8, 10], [12, 14], [16, 18]])
        self.assertTrue(mat9 + mat10 == res4)
        self.assertTrue(mat10 + mat9 == res4)

        num = 5
        mat11 = mx.Matrix(data=[1, 2])
        res5 = mx.Matrix(data=[[6, 7]])
        self.assertTrue(mat11 + num == res5)

        mat12 = mx.Matrix(data=[[1], [2]])
        res5_t = res5.transpose
        self.assertTrue(mat12 + num == res5_t)

        mat13 = mx.Matrix(data=[[1, 2, 3], [4, 5, 6]])
        res6 = mx.Matrix(data=[[6, 7, 8], [9, 10, 11]])
        self.assertTrue(num + mat13 == res6)

    def test_mat_sub(self) -> None:
        """
        Tests for substraction of two matrices
        :return:
        """
        mat1 = mx.Matrix(data=[[1, 2], [3, 4]])
        mat2 = mx.Matrix(data=[[5, 6, 7], [8, 9, 10]])
        with self.assertRaises(exs.DimensionError):
            _ = mat1 - mat2

        mat3 = mx.Matrix(data=[[1, 2, 3]])
        mat4 = mx.Matrix(data=[[4, 5, 6]])
        res1 = mx.Matrix(data=[[-3, -3, -3]])
        res2 = mx.Matrix(data=[[3, 3, 3]])
        self.assertTrue(mat3 - mat4 == res1)
        self.assertTrue(mat4 - mat3 == res2)

        mat3_t = mat3.transpose
        mat4_t = mat4.transpose
        res1_t = res1.transpose
        res2_t = res2.transpose
        self.assertTrue(mat3_t - mat4_t == res1_t)
        self.assertTrue(mat4_t - mat3_t == res2_t)

        mat5 = mx.Matrix(data=[[1, 2], [3, 4]])
        mat6 = mx.Matrix(data=[[5, 6], [7, 8]])
        res3 = mx.Matrix(data=[[-4, -4], [-4, -4]])
        res4 = mx.Matrix(data=[[4, 4], [4, 4]])
        self.assertTrue(mat5 - mat6 == res3)
        self.assertTrue(mat6 - mat5 == res4)

        mat7 = mx.Matrix(data=[[1, 2, 3], [4, 5, 6]])
        mat8 = mx.Matrix(data=[[7, 8, 9], [10, 11, 12]])
        res5 = mx.Matrix(data=[[-6, -6, -6], [-6, -6, -6]])
        res6 = mx.Matrix(data=[[6, 6, 6], [6, 6, 6]])
        self.assertTrue(mat7 - mat8 == res5)
        self.assertTrue(mat8 - mat7 == res6)

        mat9 = mx.Matrix(data=[[1, 2], [3, 4], [5, 6]])
        mat10 = mx.Matrix(data=[[7, 8], [9, 10], [11, 12]])
        res7 = mx.Matrix(data=[[-6, -6], [-6, -6], [-6, -6]])
        res8 = mx.Matrix(data=[[6, 6], [6, 6], [6, 6]])
        self.assertTrue(mat9 - mat10 == res7)
        self.assertTrue(mat10 - mat9 == res8)

        num = 5
        mat11 = mx.Matrix(data=[1, 2])
        res5 = mx.Matrix(data=[[-4, -3]])
        self.assertTrue(mat11 - num == res5)

        mat12 = mx.Matrix(data=[[1], [2]])
        res5_t = res5.transpose
        self.assertTrue(mat12 - num == res5_t)

        mat13 = mx.Matrix(data=[[1, 2, 3], [4, 5, 6]])
        res6 = mx.Matrix(data=[[-4, -3, -2], [-1, 0, 1]])
        self.assertTrue(num - mat13 == res6)

    def test_scalar_mult(self) -> None:
        """
        Tests for the scalar multiplication of matrices
        :return:
        """
        mul = 5
        # Multiplication of a matrix with just a single row
        mat1 = mx.Matrix(data=[1, 2, 3, 4])
        res1 = mat1 * mul
        mat2 = mx.Matrix(data=[[1, 2, 3, 4]])
        res2 = mat2 * mul
        self.assertListEqual(res1.data, [[5, 10, 15, 20]])
        self.assertListEqual(res1.data, res2.data)

        # Multiplication of a matrix with a single column
        mat3 = mx.Matrix(data=[[1], [2], [3], [4]])
        res3 = mat3 * mul
        self.assertListEqual(res3.data, [[5], [10], [15], [20]])

        # Multiplication of a "fuller" matrix
        mat4 = mx.Matrix(data=[[1, 2, 3], [4, 5, 6]])
        res4 = mat4 * mul
        prod = [[5, 10, 15], [20, 25, 30]]
        self.assertListEqual(res4.data, prod)

    def test_elementwise_mul(self) -> None:
        """
        Tests for the elementwise multiplication of two matrices
        :return:
        """
        # Test with incorrect shapes
        mat1 = mx.Matrix(data=[[1, 2], [3, 4]])
        mat2 = mx.Matrix(data=[[1, 2, 3], [4, 5, 6]])
        mat3 = mx.Matrix(data=[[1], [2]])
        mat4 = mx.Matrix(data=[[1, 2]])
        with self.assertRaises(exs.DimensionError):
            _ = mat1 * mat2
        with self.assertRaises(exs.DimensionError):
            _ = mat2 * mat1
        with self.assertRaises(exs.DimensionError):
            _ = mat3 * mat4
        with self.assertRaises(exs.DimensionError):
            _ = mat4 * mat3

        # Some valid cases
        mat5 = mx.Matrix(data=[[2, 2], [2, 2]])
        res1 = mx.Matrix(data=[[2, 4], [6, 8]])
        self.assertTrue(mat1 * mat5 == res1)
        self.assertTrue(mat5 * mat1 == res1)

        res2 = mx.Matrix(data=[[1], [4]])
        self.assertTrue(mat4.transpose * mat3 == res2)
        self.assertTrue(mat3 * mat4.transpose == res2)

        mat6 = mx.Matrix(data=[[2, 2, 2], [2, 2, 2]])
        res3 = mx.Matrix(data=[[2, 4, 6], [8, 10, 12]])
        self.assertTrue(mat6 * mat2 == res3)
        self.assertTrue(mat2 * mat6 == res3)

    def test_elem_exp(self) -> None:
        """
        Tests for the elementwise exponentiation
        :return:
        """
        mat1 = mx.Matrix(data=[1, 2])
        mat1 = mx.elem_exp(mat=mat1)
        res1 = [[math.exp(1), math.exp(2)]]  # Note the nested list
        self.assertListEqual(mat1.data, res1)

        mat2 = mx.Matrix(data=[[1], [2]])
        mat2 = mx.elem_exp(mat=mat2)
        res2 = [[math.exp(1)], [math.exp(2)]]
        self.assertListEqual(mat2.data, res2)

        mat3 = mx.Matrix(data=[[1, 2], [3, 4]])
        mat3 = mx.elem_exp(mat=mat3)
        res3 = [[math.exp(1), math.exp(2)], [math.exp(3), math.exp(4)]]
        self.assertListEqual(mat3.data, res3)

    def test_mat_mul(self) -> None:
        """
        Tests for the matrix multiplication
        :return:
        """
        # Error cases
        mat1 = mx.Matrix(data=[[1, 2, 3], [4, 5, 6]])
        mat2 = mx.Matrix(data=[[1, 2, 3]])
        with self.assertRaises(exs.DimensionError):
            _ = mat1 @ mat2
        with self.assertRaises(exs.DimensionError):
            _ = mat2 @ mat1
        mat3 = mx.Matrix(data=[[1], [2], [3]])
        with self.assertRaises(exs.DimensionError):
            _ = mat3 @ mat1

        # Valid cases
        prod1 = mx.Matrix(data=[[14], [32]])
        self.assertTrue(mat1 @ mat3 == prod1)
        mat4 = mx.Matrix(data=[[1, 2, 3], [4, 5, 6]])
        mat5 = mx.Matrix(data=[[1, 2], [3, 4], [5, 6]])
        prod2 = mx.Matrix(data=[[22, 28], [49, 64]])
        prod3 = mx.Matrix(data=[[9, 12, 15], [19, 26, 33], [29, 40, 51]])
        self.assertTrue(mat4 @ mat5 == prod2)
        self.assertTrue(mat5 @ mat4 == prod3)

    def test_identity(self) -> None:
        """
        Tests for the generation of identity matrices
        :return:
        """
        # Error cases
        with self.assertRaises(ValueError):
            mx.identity_matrix(n=-1)
        with self.assertRaises(ValueError):
            mx.identity_matrix(n=0)

        # Valid cases
        res1 = [[1]]
        self.assertListEqual(mx.identity_matrix(n=1).data, res1)
        res2 = [[1, 0], [0, 1]]
        self.assertListEqual(mx.identity_matrix(n=2).data, res2)
        res3 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        self.assertListEqual(mx.identity_matrix(n=4).data, res3)

    def test_mat_pow(self) -> None:
        """
        Tests for matrix power, i.e., raising a matrix to some integer
        power
        :return:
        """
        # Error case due to invalid power
        with self.assertRaises(ValueError):
            _ = mx.Matrix(data=[[1, 2], [3, 4]]) ** -1

        # Error case due to non-square matrix
        with self.assertRaises(exs.DimensionError):
            _ = mx.Matrix(data=[[1, 2, 3], [4, 5, 6]]) ** 1

        # Valid cases
        res1 = mx.Matrix(data=[[1]])
        res2 = mx.Matrix(data=[[5]])
        res3 = mx.Matrix(data=[25])  # Test also without a nested row
        self.assertTrue(mx.Matrix(data=[[5]]) ** 0 == res1)
        self.assertTrue(mx.Matrix(data=[[5]]) ** 1 == res2)
        self.assertTrue(mx.Matrix(data=[[5]]) ** 2 == res3)

        mat1 = mx.Matrix(data=[[1, 2], [3, 4]])
        mul1 = mat1 @ mat1
        self.assertTrue(mat1 ** 0 == mx.identity_matrix(n=2))
        self.assertTrue(mat1 ** 1 == mat1)
        self.assertTrue(mat1 ** 2 == mul1)
        res4 = mx.Matrix(data=[[37, 54], [81, 118]])
        self.assertTrue(mat1 ** 3 == res4)

    def test_mat_exp(self) -> None:
        """
        Tests for the matrix exponentiation
        :return:
        """
        # Error cases

        # Valid cases
        epsilon = 1e-6  # Tolerance due to floating point/rounding error
        m1 = mx.Matrix(data=[[0, -math.pi], [math.pi, 0]])
        res1 = mx.mat_exp(mat=m1)
        expected1 = mx.Matrix(data=[[-1, 0], [0, -1]])
        self.assertTrue(_almost_equal(mat1=res1, mat2=expected1, eps=epsilon))

        # Example from https://en.wikipedia.org/wiki/Matrix_exponential
        m2 = mx.Matrix(data=[[1, 4], [1, 1]])
        res2 = mx.mat_exp(mat=m2)
        t1 = (math.exp(4) + 1) / (2 * math.exp(1))
        t2 = (math.exp(4) - 1) / math.exp(1)
        t3 = (math.exp(4) - 1) / (4 * math.exp(1))
        expected2 = mx.Matrix(data=[[t1, t2], [t3, t1]])
        self.assertTrue(_almost_equal(mat1=res2, mat2=expected2, eps=epsilon))

        m3 = mx.Matrix([[-3, 0, 0], [0, 4, 0], [0, 0, 1.73]])
        res3 = mx.mat_exp(mat=m3)
        expected3 = [[math.exp(-3), 0, 0], [0, math.exp(4), 0],
                     [0, 0, math.exp(1.73)]]
        expected3 = mx.Matrix(data=expected3)
        self.assertTrue(_almost_equal(mat1=res3, mat2=expected3, eps=epsilon))

    def test_neq(self) -> None:
        """
        Tests for the negation of a matrix
        :return:
        """
        mat1 = mx.Matrix(data=[1, 2])
        mat2 = mx.Matrix(data=[-1, -2])
        self.assertTrue(-mat1 == mat2)

        mat3 = mx.Matrix(data=[[1], [2]])
        mat4 = mx.Matrix(data=[[-1], [-2]])
        self.assertTrue(mat3 == -mat4)

        mat4 = mx.Matrix(data=[[1, 2, 3], [4, 5, 6]])
        mat5 = mx.Matrix(data=[[7, 8, 9], [10, 11, 12]])
        diff = mat5 - mat4
        self.assertTrue(-mat4 + mat5 == diff)
        self.assertTrue(-mat4 - -mat5 == -mat4 + mat5)


def main() -> None:
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestMatrix))
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    main()
