from inspect import isfunction
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Union
from warnings import warn

import numpy as np
import sklearn.cluster
from sklearn.base import BaseEstimator

from denstream import micro_cluster, preprocessing
from denstream.typing import FloatArrayType, InputDict, IntArrayType, MetricsDict


class DenStream:
    """
    This class implements the DenStream algorithm:
        "Density-Based Clustering over an Evolving Data Stream with Noise" by Martin et. al. (2006).

    The notation used here are from the paper and therefore it will be easier to understand the code if ones read
        the paper first.
    """

    def __init__(
        self,
        epsilon: float,
        beta: float,
        mu: int,
        lambd: float,
        min_samples: int,
        label_metrics_list: List[Callable[[IntArrayType, IntArrayType], float]] = [],
        no_label_metrics_list: List[Callable[[FloatArrayType, IntArrayType], float]] = [],
        distance_measure: Union[None, float, Literal["fro", "nuc"]] = None,
    ):
        """
        :param epsilon: The radius used by the micro-cluster and DBScan.
        :param beta: Potential factor.
        :param mu: Weight factor used by the micro-clusters.
        :param lambd: Fading factor.
        :param min_samples: Minimum number of samples used in DBScan.
        :param label_metrics_list: List of function used to evaluate the cluster quality with labels, e.g.
            sklearn.metrics.v_measure_score. The functions requires the format function(true_labels, predicted_labels).
        :param no_label_metrics_list: List of functions used to evaluate the cluster quality without labels, e.g.
            sklearn.metrics.silhouette_score. The functions requires the format function(features, predicted_labels).
        :param distance_measure: Type of distance measure used for finding the closest p-micro-cluster.
            Need to be compatible with numpy.linalg.norm's parameter "ord".
        :return
        """

        self.epsilon = epsilon
        self.beta = beta
        self.mu = mu
        self.lambd = lambd
        self.min_samples = min_samples
        self.distance_measure = distance_measure

        self.label_metrics_list = label_metrics_list
        self.no_label_metrics_list = no_label_metrics_list
        self.metrics_results: List[Dict[str, Optional[Any]]] = []
        self._validate_init_input()

        self.Tp = (1.0 / self.lambd) * np.log((self.beta * self.mu)) / (self.beta * self.mu - 1)

        self.o_micro_clusters: List[micro_cluster.MicroCluster] = []
        self.p_micro_clusters: List[micro_cluster.MicroCluster] = []

        self.completed_o_clusters: List[micro_cluster.MicroCluster] = []
        self.completed_p_clusters: List[micro_cluster.MicroCluster] = []

        self.iterations = 0

        self.model = sklearn.cluster.DBSCAN(
            eps=self.epsilon,
            min_samples=self.min_samples,
            metric="euclidean",
            algorithm="auto",
            n_jobs=-1,
        )

    def _find_closest_cluster(self, cluster_list: List[micro_cluster.MicroCluster], feature_array: FloatArrayType) -> int:
        """
        Function for finding the closest cluster for a given point p (feature_array).

        :param cluster_list: List of micro-clusters.
        :param feature_array: array for a given data point. Must have the shape (1, num_features).
        :return: Index which specifies the closest cluster to the point p (feature_array).
        """

        cluster_centers = np.concatenate([c.center for c in cluster_list], axis=0)
        dist = np.linalg.norm(feature_array - cluster_centers, axis=1, ord=self.distance_measure)
        closest_cluster_index = np.argmin(dist)
        return int(closest_cluster_index)

    def _calculate_xi(self, time: int, creation_time: int) -> float:
        """
        Function for calculating the xi-value (see the paper for further context).

        :param time: Specifying the current time.
        :param creation_time: Specifying the creation time of a cluster.
        :return: The xi value.
        """

        xi: float = (np.power(2, -self.lambd * (time - creation_time + self.Tp)) - 1) / (np.power(2, -self.lambd * self.Tp) - 1)
        return xi

    def _merging(self, current_time: int, feature_array: FloatArrayType, label: Optional[int] = None) -> None:
        """
        The merging step of a point p (feature_array) as described in the paper.

        :param current_time: The current time.
        :param feature_array: Array for a given data point p.
        :param label: Specifying the true label of a data point. None indicates that it is not provided.
        :return
        """

        if len(self.p_micro_clusters) > 0:
            closest_p_index = self._find_closest_cluster(self.p_micro_clusters, feature_array)
            closest_p_cluster = self.p_micro_clusters[closest_p_index]

            closest_p_cluster.append(current_time, feature_array, label)
            r_p, weight, cf1 = closest_p_cluster.calculate_radius(current_time)

            if r_p <= self.epsilon:
                closest_p_cluster.update_parameters(cf1_score=cf1, weight=weight)
                return
            else:
                closest_p_cluster.pop()

        if len(self.o_micro_clusters) > 0:
            closest_o_index = self._find_closest_cluster(self.o_micro_clusters, feature_array)
            closest_o_cluster = self.o_micro_clusters[closest_o_index]

            closest_o_cluster.append(current_time, feature_array, label)
            r_o, weight, cf1 = closest_o_cluster.calculate_radius(current_time)

            if r_o <= self.epsilon:
                closest_o_cluster.update_parameters(cf1_score=cf1, weight=weight)

                if closest_o_cluster.weight > self.beta * self.mu:
                    self.p_micro_clusters.append(closest_o_cluster)
                    self.o_micro_clusters.pop(closest_o_index)
                    return
            else:
                # The clusters is not compact enough, therefore removing the newly added point.
                closest_o_cluster.pop()

        new_o_cluster = micro_cluster.MicroCluster(current_time, self.lambd)
        new_o_cluster.append(current_time, feature_array, label)
        new_o_cluster.update_parameters(time=current_time)
        self.o_micro_clusters.append(new_o_cluster)

    def _prune_p_clusters(self, time: int) -> None:
        """
        Pruning the potential activate clusters.

        :param time: The current time.
        :return
        """

        for idx in range(len(self.p_micro_clusters) - 1, -1, -1):
            p_cluster = self.p_micro_clusters[idx]
            p_cluster.update_parameters(time=time)

            if p_cluster.weight < self.beta * self.mu:
                self.completed_p_clusters.append(p_cluster)
                self.p_micro_clusters.pop(idx)

    def _prune_o_clusters(self, time: int) -> None:
        """
        Pruning the outlier activate clusters.

        :param time: The current time.
        :return
        """

        for idx in range(len(self.o_micro_clusters) - 1, -1, -1):
            o_cluster = self.o_micro_clusters[idx]
            o_cluster.update_parameters(time=time)
            xi = self._calculate_xi(time, o_cluster.creation_time)

            if o_cluster.weight < xi:
                self.completed_o_clusters.append(o_cluster)
                self.o_micro_clusters.pop(idx)

    def partial_fit(
        self,
        feature_array: FloatArrayType,
        time: int,
        label: Optional[int] = None,
        request_period: Optional[Any] = None,
    ) -> None:
        """
        :param feature_array: Array for a given data point p.
        :param time: The current time.
        :param label: Specifying the true label of a data point. None indicates that it is not provided.
        :param request_period: Specifying when (in terms of #data-points) we should compute the clusters.
            It can have the types:
                - An integer, i.e. do the clustering every request_period.
                - List of integers, i.e. cluster if the iteration number is request_period[idx].
                - None, i.e. do no cluster with self.model.
        :return
        """

        DenStream._validate_fit_input(time, feature_array, label)
        self.iterations += 1

        self._merging(time, feature_array, label)

        if time % np.ceil(self.Tp) == 0:
            self._prune_p_clusters(time)
            self._prune_o_clusters(time)

        if isinstance(request_period, int):
            if self.iterations % request_period == 0:
                self._cluster_evaluate(self.iterations)
        elif isinstance(request_period, list):
            if self.iterations in request_period:
                self._cluster_evaluate(self.iterations)

    def fit_generator(
        self,
        generator: Iterator[InputDict],
        normalize: bool = False,
        request_period: Optional[Any] = None,
        warmup_period: int = 1,
    ) -> None:
        """
        Fitting DenStream to a stream of data-points (i.e. python generator).
            It will run until the generator does not have any data points left.

        :param generator: used to stream data-points to the model. It must yield a python dictionary with the keys:
            time [int]: integer for the time when the data point arrived.
            feature_array [np.ndarray]: numpy array for the data point. Must have the shape (1, num_features).
            label [Optional[int]]: the true label of the data point. Needed for self.label_metrics_list.
        :param normalize: Whether to normalize the features to zero mean and unit variance.
            The normalization is done with rolling statistics, i.e. update mean and variance iterable.
        :param request_period: Specifying when (in terms of #data-points) we should compute the clusters.
            It can have the types:
                - An integer, i.e. do the clustering every request_period.
                - List of integers, i.e. cluster if the iteration number is request_period[idx].
                - None, i.e. do no cluster with self.model.
        :param warmup_period: The number of samples used to "warm-up" the rolling mean and variance, if normalize=True.
        :return
        """

        if self.iterations > 0:
            raise RuntimeError("Seems like the method as already been fitted, try to re-create it.")

        if normalize:
            for _ in range(warmup_period):
                try:
                    gen_dict = generator.__next__()
                except StopIteration:
                    raise RuntimeError(f"Not enough samples where given for the warmup-period, warmup_period={warmup_period}")

                feature_array = gen_dict["feature_array"]
                rs = preprocessing.RollingStats(feature_array.shape)
                rs.update_statistics(feature_array)

        while True:
            try:
                gen_dict = generator.__next__()
            except StopIteration:
                break
            time = gen_dict["time"]
            feature_array = gen_dict["feature_array"]

            if "label" in gen_dict:
                label = gen_dict["label"]
            else:
                label = None

            if normalize:
                DenStream._validate_fit_input(time, feature_array, label)
                rs.update_statistics(feature_array)
                feature_array = rs.normalize(feature_array)

            self.partial_fit(feature_array, time, label, request_period=request_period)

    def set_clustering_model(self, new_model: BaseEstimator) -> None:
        """
        This method allows the user to use another clustering method than DBScan, e.g. K-Means.
        Note that it needs to be a sklearn model.
        Example usage:
            new_model = sklearn.cluster.KMeans(n_clusters=2)
            DenStream.set_clustering_model(new_model)

        :param new_model: A sklearn clustering model.
        :return:
        """

        if not isinstance(new_model, BaseEstimator):
            raise ValueError("The new model needs to be a sklearn-model.")

        self.model = new_model

    def _cluster_evaluate(self, iteration: int) -> None:
        """
        Calling request clustering and computing the metrics.

        :param iteration: current iteration, i.e. #data-points.
        :return
        """

        predicted_labels = self._request_clustering()

        if len(predicted_labels) > 0:
            metrics = []

            if len(self.label_metrics_list) > 0:
                metrics += self._compute_label_metrics(predicted_labels)
            if len(self.no_label_metrics_list) > 0:
                # Checking that we have atleast two clusters (exluding outlier clusters, i.e. label=-1).
                if len(set(predicted_labels[predicted_labels != -1])) > 1:
                    metrics += self._compute_no_label_metric(predicted_labels)
                else:
                    warn("Number of predicted clusters are 1 or less. Therefore no-label-metrics are not computed!")
            if len(metrics) > 0:
                self.metrics_results.append({"iteration": iteration, "metrics": metrics})
        else:
            self.metrics_results.append({"iteration": iteration, "metrics": None})

    def _request_clustering(self) -> FloatArrayType:
        """
        Clustering based on self.model for the p-micro-clusters.

        :return: Array of predicted labels for each p-micro-cluster.
        """

        if len(self.p_micro_clusters) > 0:
            center_array = np.concatenate([c.center for c in self.p_micro_clusters], axis=0)
        else:
            return np.empty(0, dtype=np.float32)

        # TODO: Should the new clusters be connected? I.e. if micro-cluster 1 and 2 and connected, should they be merged
        local_model = sklearn.base.clone(self.model)
        predicted_labels: FloatArrayType = local_model.fit_predict(center_array)
        return predicted_labels

    def _compute_label_metrics(self, predicted_labels: FloatArrayType) -> List[MetricsDict]:
        """
        Compute the label metrics given the predicted labels.

        :param predicted_labels: Array of the predicted labels for each p-micro-cluster.
        :return: List of dictionaries with the values for each label metrics.
            It has the key name (i.e. name of the metric) and value (i.e. the value of the metric).
        """
        predicted_list, true_list = [], []

        for idx, predicted_label in enumerate(predicted_labels):
            true_labels = self.p_micro_clusters[idx].labels_array
            true_list.append(np.array(true_labels))

            repeated_prediction = np.repeat(predicted_label, len(true_labels))
            predicted_list.append(repeated_prediction)

        true_array = np.concatenate(true_list, axis=0)
        predicted_array = np.concatenate(predicted_list, axis=0)

        results = []
        for metric in self.label_metrics_list:
            val = metric(true_array, predicted_array)
            result_dict = MetricsDict(name=metric.__name__, value=val)
            results.append(result_dict)
        return results

    def _compute_no_label_metric(self, predicted_labels: FloatArrayType) -> List[MetricsDict]:
        """
        Compute the no-label metrics given the predicted labels.

        :param predicted_labels: Array of the predicted labels for each p-micro-cluster.
        :return: List of dictionaries with the values for each no-label metrics.
            It has the key name (i.e. name of the metric) and value (i.e. the value of the metric).
        """
        predicted_list, feature_list = [], []

        for idx, predicted_label in enumerate(predicted_labels):
            features = self.p_micro_clusters[idx].features_array
            feature_list.append(np.array(features))

            repeated_prediction = np.repeat(predicted_label, len(features))
            predicted_list.append(repeated_prediction)

        combined_feature_array = np.concatenate(feature_list, axis=0)
        predicted_array = np.concatenate(predicted_list, axis=0)

        results = []
        for metric in self.no_label_metrics_list:
            val = metric(combined_feature_array, predicted_array)
            result_dict = MetricsDict(name=metric.__name__, value=val)
            results.append(result_dict)
        return results

    def _validate_init_input(self) -> None:
        """
        Checking that the input to init is valid.
        :return
        """

        if isinstance(self.epsilon, int) or isinstance(self.epsilon, float):
            if self.epsilon <= 0:
                raise ValueError("epsilon must be positive.")
        else:
            raise ValueError("epsion must be of type float or integer.")

        if isinstance(self.beta, float):
            if not 0.0 < self.beta <= 1.0:
                raise ValueError("beta must be between 0.0 and 1.0.")
        else:
            raise ValueError("beta must be of type float")

        if isinstance(self.mu, int):
            if self.mu <= 0:
                raise ValueError("mu must be positive.")
        else:
            raise ValueError("mu must be of type integer.")

        if isinstance(self.min_samples, int):
            if self.min_samples <= 0:
                raise ValueError("min_samples must be positive.")
        else:
            raise ValueError("min_samples must be of type integer.")

        if isinstance(self.lambd, int) or isinstance(self.lambd, float):
            if self.min_samples <= 0.0:
                raise ValueError("lambd must be positive.")
        else:
            raise ValueError("lambd must be of type float or integer.")

        if self.beta * self.mu <= 1.0:
            raise ValueError("beta * mu <= 1.0 which will cause problems when computing Tp.")

        for label_metric in self.label_metrics_list:
            if not isfunction(label_metric):
                raise ValueError("The label metric input(s) must be a function.")

        for no_label_metric in self.no_label_metrics_list:
            if not isfunction(no_label_metric):
                raise ValueError("The no-label metric input(s) must be a function.")

    @staticmethod
    def _validate_fit_input(time: int, feature_array: FloatArrayType, label: Optional[int] = None) -> None:
        """
        Validate the fit_generator's input parameters.

        :param time: The current time.
        :param feature_array: Array for a given data point p.
        :param label: Specifying the true label of a data point. None indicates that the label is not provided.
        :return
        """

        if not isinstance(feature_array, np.ndarray):
            raise ValueError(f"Provided x is not an numpy.ndarray, type(x)={type(feature_array)}")
        elif len(feature_array.shape) != 2:
            raise ValueError(f"feature_array need to have the shape (1, num_features), " f"given shape={feature_array.shape}")

        if not isinstance(time, int):
            raise ValueError(f"Provided time is not an int. type(time)={type(time)}")
        elif time < 0:
            raise ValueError(f"Time needs to be positive. time={time}")

        if not isinstance(label, int) and label is not None:
            raise ValueError(f"Provided label is not an int or None. label={label}")
