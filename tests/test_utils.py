import unittest

import numpy as np

from denstream import utils


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.TOL = 1e-8
        self.SEED = 42

    def test_fading_function(self):
        """
        This test is designed to check that the fading function works as expected,
            both for scalars and vectors (numpy arrays).
        """

        # Testing fading function for scalar values
        lambd = 1.0
        time = 2
        assert np.abs(utils.fading_function(lambd, time) - 0.25) < self.TOL

        # Testing fading function for numpy arrays
        lambd_array = np.repeat(lambd, 4).reshape((4, 1))
        time_array = np.array([1, 2, 3, 4]).reshape((4, 1))

        expected_array = np.array([0.5, 0.25, 0.125, 0.0625]).reshape((4, 1))
        actual_array = utils.fading_function(lambd_array, time_array)

        self.assertTrue(np.linalg.norm(actual_array - expected_array) < self.TOL)

    def test_cf1_calculations(self):
        """
        This test checks that the calculation of the CF1-score is the same for the numpy and the numba version.
        """

        np.random.seed(self.SEED)
        x = np.random.uniform(0, 1, size=(100, 2))
        fading_array = np.random.uniform(0, 1, size=(100, 1))

        np_cf1 = utils.numpy_cf1(x, fading_array)
        numba_cf1 = utils.numba_cf1(x, fading_array)

        self.assertTrue(np.linalg.norm(np_cf1 - numba_cf1) < self.TOL)

    def test_cf2_calculations(self):
        """
        This test checks that the calculation of the CF2-score is the same for the numpy and the numba version.
        """

        np.random.seed(self.SEED)
        x = np.random.uniform(0, 1, size=(100, 2))
        fading_array = np.random.uniform(0, 1, size=(100, 1))

        np_cf2 = utils.numpy_cf2(x, fading_array)
        numba_cf2 = utils.numba_cf2(x, fading_array)

        self.assertTrue(np.linalg.norm(np_cf2 - numba_cf2) < self.TOL)


if __name__ == "__main__":
    unittest.main()
