from typing import Optional, Tuple

import numpy as np
from typing_extensions import Unpack

from denstream import utils
from denstream.typing import FloatArrayType, UpdateParameters


class MicroCluster:
    """
    This class represents the micro-potential / outlier cluster describe in the paper:
        "Density-Based Clustering over an Evolving Data Stream with Noise" by Martin Ester et. al.
    """

    def __init__(self, creation_time: int, lambd: float):
        """
        Initializing the micro-cluster.

        :param creation_time: The creation time (i.e. "now") for this micro-cluster.
        :param lambd: Fading factor for this cluster.
        :return
        """

        self.lambd = lambd
        self.creation_time = creation_time

        self.features_array = np.array([], dtype=np.float32)
        self.time_array = np.array([], dtype=np.int32)
        self.labels_array = np.array([], dtype=np.int32)

        self.weight = np.array(np.nan, dtype=np.float32)
        self.center = np.array([], dtype=np.float32)

        self.cf1_func = utils.numba_cf1
        self.cf2_func = utils.numba_cf2

    def append(self, time: int, feature_array: FloatArrayType, label: Optional[int] = None) -> None:
        """
        This function appends data-points to the features / time / labels arrays.

        :param time: The time to append.
        :param feature_array: Array for a given data-point. Must have the shape (1, num_features).
        :param label: Specifying the true cluster label of a data-point. None indicates that it is not given.
        :return
        """

        time_array = np.array(time).reshape((1, 1))

        assert len(self.features_array) == len(self.time_array)
        if len(self.features_array) == 0:
            self.features_array = feature_array
            self.time_array = time_array
        else:
            self.features_array = np.append(self.features_array, feature_array, axis=0)
            self.time_array = np.append(self.time_array, time_array, axis=0)

        if label is not None:
            label_array = np.array([label]).reshape((1,))

            if len(self.labels_array) == 0:
                self.labels_array = label_array
            else:
                self.labels_array = np.append(self.labels_array, label_array, axis=0)

    def pop(self) -> None:
        """
        This function pops out the last data-point (i.e. the len(features_array) -1 element).

        :return
        """

        assert len(self.features_array) == len(self.time_array)
        if len(self.features_array) == 0:
            pass
        else:
            self.features_array = np.delete(self.features_array, [len(self.features_array) - 1], axis=0)
            self.time_array = np.delete(self.time_array, [len(self.time_array) - 1], axis=0)

        if len(self.labels_array) == 0:
            pass
        else:
            self.labels_array = np.delete(self.labels_array, [len(self.labels_array) - 1], axis=0)

    def _calculate_fading(self, time: int) -> FloatArrayType:
        """
        This function calculates the fading values for time for this micro-cluster.

        :param time: The time value for which to compute the fading value for.
        :return: Array containing the fading values from this micro-cluster.
        """

        return utils.fading_function(self.lambd, time - self.time_array)

    def calculate_radius(self, time: int) -> Tuple[FloatArrayType, FloatArrayType, FloatArrayType]:
        """
        Calculating the radius of a micro-cluster according to the paper
            https://archive.siam.org/meetings/sdm06/proceedings/030caof.pdf.

        :param time: Time value used for calculating the radius.
        :return: Calculate radius, weight and CF1-score.
        """

        fading_array = utils.fading_function(self.lambd, time - self.time_array)
        weight = np.sum(fading_array, axis=0)
        cf1 = self.cf1_func(self.features_array, fading_array)
        cf2 = self.cf2_func(self.features_array, fading_array)

        radius_squared = np.sum(np.abs(cf2), axis=1) / weight - 1 / np.power(weight, 2) * np.dot(cf1, cf1.T)
        radius_squared = radius_squared if radius_squared > 0 else 0
        radius = np.sqrt(radius_squared)

        return radius, weight, cf1

    def update_parameters(self, **kwargs: Unpack[UpdateParameters]) -> None:
        """
        Updating the weight and center parameter for the micro-cluster.
        There is two modes:
            - One when only "time" is given. Then calculate the weight and cf1 from scratch.
            - If "cf1_score" and "weight" is given, simply use them. This is done to avoid recomputing them.

        :param kwargs:
        :return
        """

        if "time" in kwargs:
            fading_array = utils.fading_function(self.lambd, kwargs["time"] - self.time_array)
            weight = np.sum(fading_array, axis=0)
            self.weight = weight
            self.center = self.cf1_func(self.features_array, fading_array) / weight
        elif "cf1_score" in kwargs and "weight" in kwargs:
            self.center = kwargs["cf1_score"] / kwargs["weight"]
            self.weight = kwargs["weight"]
        else:
            raise ValueError("Wrong input to MicroCluster.update_parameters")
