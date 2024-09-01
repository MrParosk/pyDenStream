import numpy as np
from numba import jit

from denstream.typing import FloatArrayType, IntArrayType


def fading_function(lambd: float, time: IntArrayType) -> FloatArrayType:
    """
    Calculating the fading value.

    :param lambd: Fading factor.
    :param time: Specifying the time.
    :return: The calculated fading array.
    """

    return np.power(2, -lambd * time, dtype=np.float32)


def numpy_cf1(x: FloatArrayType, fading_array: FloatArrayType) -> FloatArrayType:
    """
    Calculating the CF1 according to the paper https://archive.siam.org/meetings/sdm06/proceedings/030caof.pdf,
        using numpy.

    :param x: Array containing the data points.
    :param fading_array: Array containing the calculated fading values.
    :return: Array with the calculated CF1 values.
    """

    x_weighted = np.multiply(x, fading_array)
    x_sum: FloatArrayType = np.sum(x_weighted, axis=0)
    x_sum = x_sum.reshape(1, x.shape[1])
    return x_sum


@jit(nopython=True, cache=True)  # type: ignore
def numba_cf1(x: FloatArrayType, fading_array: FloatArrayType) -> FloatArrayType:
    """
    Calculating the CF1 according to the paper https://archive.siam.org/meetings/sdm06/proceedings/030caof.pdf,
        using numba.

    :param x: Array containing the data points.
    :param fading_array: Array containing the calculated fading values.
    :return: Array with the calculated CF1 values.
    """

    return_array = np.zeros((1, x.shape[1]), dtype=fading_array.dtype)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            return_array[0, j] += x[i, j] * fading_array[i, 0]

    return return_array.reshape((1, x.shape[1]))


def numpy_cf2(x: FloatArrayType, fading_array: FloatArrayType) -> FloatArrayType:
    """
    Calculating the CF2 according to the paper https://archive.siam.org/meetings/sdm06/proceedings/030caof.pdf,
        using numpy.

    :param x: Array containing the data points.
    :param fading_array: Array containing the calculated fading values.
    :return: Array with the calculated CF2 values.
    """

    x_squared = np.multiply(x, x)
    x_weighted = np.multiply(x_squared, fading_array)
    x_sum: FloatArrayType = np.sum(x_weighted, axis=0)
    return x_sum.reshape((1, x.shape[1]))


@jit(nopython=True, cache=True)  # type: ignore
def numba_cf2(x: FloatArrayType, fading_array: FloatArrayType) -> FloatArrayType:
    """
    Calculating the CF2 according to the paper https://archive.siam.org/meetings/sdm06/proceedings/030caof.pdf,
        using numba.

    :param x: Array containing the data points.
    :param fading_array: Array containing the calculated fading values.
    :return: Array with the calculated CF2 values.
    """

    return_array = np.zeros((1, x.shape[1]), dtype=x.dtype)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            return_array[0, j] += x[i, j] * x[i, j] * fading_array[i, 0]

    return return_array.reshape((1, x.shape[1]))
