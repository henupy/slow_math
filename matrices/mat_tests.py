"""
Unittests for the matrix operations
"""

import unittest
import operations

class TestMatOps(unittest.TestCase):
    pass


def main() -> None:
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestMatOps))
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    main()
