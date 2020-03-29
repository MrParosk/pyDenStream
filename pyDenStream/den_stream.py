import numpy as np
import sklearn
from typing import List, Optional, Callable, Any
from warnings import warn

from pyDenStream import utils
from pyDenStream import micro_cluster
from pyDenStream import preprocessing


class DenStream:
    """
    This class implements the DenStream algorithm:
        "Density-Based Clustering over an Evolving Data Stream with Noise" by Martin et. al. (2006).

    The notation used here are from the paper and therefore it will be easier to understand the code if ones read
        the paper first.
    """

    def __init__(self, epsilon: float, beta: float, mu: int, lambd: float, min_samples: int,
                 label_metrics_list: List[Callable] = [], unlabel_metrics_list: List[Callable] = [],
                 distance_measure: Optional[str] = None):
        """
        :param epsilon: The radius used by the micro-cluster and DBScan.
        :param beta: Potential factor.
        :param mu: Weight factor used by the micro-clusters.
        :param lambd: Fading factor.
        :param min_samples: Minimum number of samples used in DBScan.
        :param label_metrics_list: List of function used to evaluate the cluster quality with labels,
            e.g. sklearn.metrics.v_measure_score. The function needs to be of the format function(true_labels, predicted_labels).
        :param unlabel_metrics_list: List of function used to evaluate the cluster quality without labels,
            e.g. sklearn.metrics.silhouette_score. The function needs to be of the format function(features, predicted_labels).
        :param distance_measure: Type of distance measure used for finding the closest p-micro-cluster.
            Need to be compatible with numpy.linalg.norm's parameter "ord".
        """

        self.epsilon = epsilon
        self.beta = beta
        self.mu = mu
        self.min_samples = min_samples
        self.distance_measure = distance_measure


        self._validate_init_input()
        self.Tp = (1.0 / self.lambd) * np.log((self.beta * self.mu)) / (self.beta * self.mu - 1)

        self.o_micro_clusters = []
        self.p_micro_clusters = []

        self.completed_o_clusters = []
        self.completed_p_clusters = []

        self.metrics_results = []
        self.label_metrics_list = label_metrics_list
        self.unlabel_metrics_list = unlabel_metrics_list

        self.model = sklearn.cluster.DBSCAN(eps=self.epsilon,
                                            min_samples=self.min_samples,
                                            metrics="euclidean",
                                            algorithm="auto",
                                            n_jobs=-1)

    def _find_closest_cluster(self, cluster_list: List[micro_cluster.MicroCluster], feature_array: np.ndarray) -> int:
        """
        Function for finding the closest cluster for a given point p (feature_array).

        :param cluster_list: List of micro-clusters.
        :param feature_array: array for a given data point. Must have the shape (1, num_features).
        :return: Index which specifies the closest cluster to the point p (feature_array).
        """

        cluster_centers = np.concatenate([c.center for c in cluster_list], axis=0)
        dist = np.linalg.norm(feature_array - cluster_centers, axis=1, ord=self.distance_measure)
        closest_cluster_index = np.argmin(dist)
        return closest_cluster_index

    def _calculate_xi(self, time: int, creation_time: int) -> float:
        """
        Function for calculating the xi-value (see the paper for further context).

        :param time: Specifying the current time.
        :param creation_time: Specifying the creation time of a cluster.
        :return: The xi value.
        """

        xi = (np.power(2, - self.lambd * (time - creation_time + self.Tp)) -1) \
        / (np.power(2, -self.lamd * self.Tp) - 1)
        return xi

    def _merging(self, current_time: int, feature_array: np.ndarray, label: Optional[int] = None) -> None:
        """
        The merging step of a point p (feature_array) as described in the paper.

        :param current_time: The current time.
        :param feature_array: Array for a given data point p.
        :param label: Specifying the true label of a data point. None indicates that it is not provided.
        """

        if len(self.p_micro_clusters) > 0:
            closest_p_index = self._find_closest_cluster(self.o_micro_clusters, feature_array)
            closest_p_cluster = self.p_micro_clusters[closest_p_index]

            closest_p_cluster.append(current_time, feature_array, label)
            r_p, weight, cf1 = closest_p_cluster.calulcate_radius(current_time)

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
        """

        for idx in range(len(self.o_micro_clusters) - 1, -1, -1):
            o_cluster = self.o_micro_clusters[idx]
            o_cluster.update_parameters(time=time)
            xi = self._calculate_xi(time, o_cluster.creation_time)

            if o_cluster.weight < xi:
                self.completed_o_clusters.append(o_cluster)
                self.o_micro_clusters.pop(idx)

    def fit_generator(self, generator, normalize: bool=False, request_period: Optional[Any] = None) -> None:
        """
        Fitting DenStream to a stream of data-points (i.e. python generator).
            It will run until the generator does not have any data points left.

        :param generator: used to stream data-points to the model. It must yield a python dictionary with the keys:
            time [int]: integer for the time when the data point arrived.
            feature_array [np.ndarray]: numpy array for the data point. Must have the shape (1, num_features).
            label [Optional[int]]: the true label of the data point. Needed for self.label_metrics_list.
        :param normalize: Whether to normalize the features to zero mean and unit variance.
            The normalization is done with rolling statistics, i.e. update mean and variance iterable.
        :param request_period: Specifying when (in terms of #data-points) we should compute the clusters. It can have the types:
            - An integer, i.e. do the clustering every request_period.
            - List of integers, i.e. cluster if the iteration number is request_period[idx].
            - None, i.e. do no cluster with self.model.
        """

        if normalize:
            try:
                gen_dict = generator.__next__()
            except StopIteration:
                raise ValueError("Given generator was empty")

            feature_array = gen_dict["feature_array"]
            rs = preprocessing.RollingStats(feature_array.shape)
            rs.update_statistics(feature_array)

        iterations = 0
        while True:
            iterations += 1

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

            self._validate_fit_input(time, feature_array, label)

            if normalize:
                rs.update_statistics(feature_array)
                feature_array = rs.normalize(feature_array)

            self._merging(time, feature_array, label)

            if time % np.ceil(self.Tp) == 0:
                self._prune_p_clusters(time)
                self._prune_o_clusters(time)

            if isinstance(request_period, int):
                if iterations % request_period == 0:
                    self._cluster_evaluate(iterations)
            elif isinstance(request_period, list):
                if iterations in request_period:
                    self._cluster_evaluate(iterations)

    def _cluster_evaluate(self, iteration: int) -> None:
        """
        Calling request clustering and computing the metrics.

        :param iteration: current iteration, i.e. #data-points.
        """

        predicted_labels = self._request_clustering()

        if len(predicted_labels) > 0:
            metrics = []

            if len(self.label_metrics_list) > 0:
                metrics += self._compute_label_metrics(predicted_labels)
            if len(self.unlabel_metrics_list) > 0:
                # Checking that we have atleast two clusters (exluding outlier clusters, i.e. label=-1).
                if len(set(predicted_labels[predicted_labels != -1])) > 1:
                    metrics += self._compute_unlabel_metric(predicted_labels)
                else:
                    warn(f"Number of predicted clusters are 1 or less. Therefore unlabel metrics are not computed!")
            if len(metrics) > 0:
                self.metrics_results.append({"iteration": iteration, "metrics": metrics})
        else:
            self.metrics_results.append({"iteration": iteration, "metrics": None})

    def _request_clustering(self) -> np.ndarray:
        """
        Clustering based on self.model for the p-micro-clusters.

        :return: Array of predicted labels for each p-micro-cluster.
        """

        if len(self.p_micro_clusters) > 0:
            center_array = np.concatenate([c.center for c in self.p_micro_clusters], axis=0)
        else:
            return np.ndarray([])

        # TODO: Should the new clusters be connected? I.e. if micro-cluster 1 and 2 and connected, should they be merged?
        local_model = sklearn.base.clone(self.model)
        predicted_labels = local_model.fit_predict(center_array)
        return predicted_labels

    def _compute_label_metric(self, predicted_labels: np.ndarray) -> List:
        """
        Compute the label metrics given the predicted labels.

        :param predicted_labels: Array of the predicted labels for each p-micro-cluster.
        :return: List of dictionaries with the values for each label metrics.
            It has the key name (i.e. name of the metric) and value (i.e. the value of the metric).
        """
        pass

    def _compute_unlabel_metric(self, predicted_labels: np.ndarray) -> List:
        """
        :param predicted_labels:
        :return:
        """
        pass

    def _validate_fit_input(self, time: int, feature_array: np.ndarray, label: Optional[int] = None):
        pass

    def _validate_init_input(self):
        pass
