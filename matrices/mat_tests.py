"""
Unittests for the matrix operations
"""

import unittest
import operations as ops


class TestMatOps(unittest.TestCase):
    def test_scalar_mult(self) -> None:
        """
        Tests for the scalar multiplication of matrices. Test for a non-scalar
        multiplier is left out since the correct type for it is type hinted.
        :return:
        """
        mul = 5
        # Multiplication of a matrix with just a single row
        mat1 = [1, 2, 3, 4]
        self.assertListEqual(ops.scalar_mult(mat1, mul), [5, 10, 15, 20])

        # Multiplication of a matrix with a single column
        mat2 = [[1], [2], [3], [4]]
        self.assertListEqual(ops.scalar_mult(mat2, mul), [[5], [10], [15], [20]])

        # Multiplication of empty matrices
        self.assertListEqual(ops.scalar_mult([], mul), [])
        self.assertListEqual(ops.scalar_mult([[]], mul), [[]])

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
