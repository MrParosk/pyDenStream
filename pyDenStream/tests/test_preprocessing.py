import numpy as np
from pyDenStream import preprocessing


TOL = 1e-6


def test_rolling_stats():
    """
    This test is designed to check that the rolling statistics (i.e. mean and variance) works as expected.
    """

    x1 = np.array([100, 1]).reshape((1, 2))
    x2 = np.array([200, 2]).reshape((1, 2))

    rs = preprocessing.RollingStats((1, 2))
    rs.update_statistics(x1)
    rs.update_statistics(x2)

    expected_mean = np.array([150, 1.5]).reshape((1, 2))
    expected_variance = np.array([2.5e3, 2.5e-1]).reshape((1, 2))

    assert(np.linalg.norm(rs.mean - expected_mean) < TOL)
    assert(np.linalg.norm(rs.variance - expected_variance) < TOL)