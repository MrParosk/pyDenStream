from typing import Tuple

import numpy as np

from denstream.typing import FloatArrayType


class RollingStats:
    """
    This class implements rolling statistics, i.e. mean and variance are updated for each new data-point.
    """

    def __init__(self, dim: Tuple[int, ...], eps: float = 1e-10):
        """
        Initializing the rolling statistics class.

        :param dim - describing the dimension of input data, .e.g. (1, 5).
        :param eps: Constant rensuring we don't divide by zero.
        :return
        """

        self.dim = dim
        self.mean = np.zeros(self.dim)

        self.variance = np.zeros(self.dim)
        self.sse = np.zeros(self.dim)

        self.num_data_points = 0
        self.eps = eps

    def update_statistics(self, x: FloatArrayType) -> None:
        """
        Updating the mean and variance according to x. The  update equations can be found here:
            https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance.

        :param x: Array with dimension equal to self.dim.
        :return:
        """

        self.num_data_points += 1

        old_mean = self.mean
        self.mean = self.mean + (x - self.mean) / self.num_data_points
        self.mean = self.mean.reshape(self.dim)

        self.sse = self.sse + np.multiply(x - old_mean, x - self.mean)
        self.sse = self.sse.reshape(self.dim)
        self.variance = self.sse / self.num_data_points

    def normalize(self, x: FloatArrayType) -> FloatArrayType:
        """
        Normalizing the input data.

        :param x: Input array.
        :return: Normalized input array.
        """

        return (x - self.mean) / (np.sqrt(self.variance) + self.eps)
