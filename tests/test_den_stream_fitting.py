import unittest

import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans

from denstream.den_stream import DenStream
from denstream.micro_cluster import MicroCluster

from .test_helpers import generate_test_data


class TestDenStreamFitting(unittest.TestCase):
    def setUp(self):
        self.TOL = 1e-6

    def test_fit_generator_cluster(self):
        """
        This test is designed to check that the expected number of p/o-micro-clusters are formed from a stream (generator).
        The input have two expected p-micro-clusters and three expected o-micro-clusters.
        """

        x_inputs = np.array(
            [
                [4.0, 4.0],
                [-4.0, -4.0],
                [3.99, 3.99],
                [-10.0, -10.0],
                [4.01, 4.01],
                [-4.01, -4.01],
                [300.0, 300.0],
                [10.0, -10.0],
            ]
        )

        time_input = [1, 1, 1, 1, 1, 1, 1, 1]

        def generator(feature_arrays, time_list):
            for i in range(0, len(time_input)):
                yield {
                    "time": time_list[i],
                    "feature_array": feature_arrays[i, :].reshape((1, 2)),
                }

        eps = 1
        lambd = 1
        beta = 0.5
        mu = 3
        min_samples = 3

        ds = DenStream(eps, beta, mu, lambd, min_samples)
        gen = generator(x_inputs, time_input)

        ds.fit_generator(gen)

        self.assertEqual(len(ds.o_micro_clusters), 3)
        self.assertEqual(len(ds.p_micro_clusters), 2)
        self.assertEqual(len(ds.completed_o_clusters), 0)
        self.assertEqual(len(ds.completed_p_clusters), 0)

    def test_fit_generator_fading(self):
        """
        This test is designed to check that the micro-clusters are fading, i.e. the activate micro-clusters are moved
            to the completed ones.
        """

        x_inputs = np.array(
            [
                [-4.0, -4.0],
                [4.0, 4.0],
                [3.99, 3.99],
                [-10.0, 10.0],
                [4.01, 4.01],
                [-4.01, -4.01],
                [300.0, 300.0],
                [10.0, -10.0],
            ]
        )

        time_input = [4, 1, 1, 1, 1, 4, 1, 4]

        def generator(feature_arrays, time_list):
            for i in range(0, len(time_input)):
                print(i)
                yield {
                    "time": time_list[i],
                    "feature_array": feature_arrays[i, :].reshape((1, 2)),
                }

        eps = 1
        lambd = 1
        beta = 0.9
        mu = 2
        min_samples = 2

        ds = DenStream(eps, beta, mu, lambd, min_samples)
        gen = generator(x_inputs, time_input)

        ds.fit_generator(gen)

        # Asserting that the activate p/o-micro-clusters have the expected size/number of data points.
        self.assertEqual(len(ds.o_micro_clusters), 1)
        self.assertEqual(len(ds.o_micro_clusters[0].features_array), 1)
        self.assertEqual(len(ds.o_micro_clusters[0].time_array), 1)

        self.assertEqual(len(ds.p_micro_clusters), 1)
        self.assertEqual(len(ds.p_micro_clusters[0].features_array), 2)
        self.assertEqual(len(ds.p_micro_clusters[0].time_array), 2)

        # Checking that the inactivate p/o-micro-clusters have the expected size / number of data-points.
        self.assertEqual(len(ds.completed_p_clusters), 1)
        self.assertEqual(len(ds.completed_p_clusters[0].features_array), 3)
        self.assertEqual(len(ds.completed_p_clusters[0].time_array), 3)

        self.assertEqual(len(ds.completed_o_clusters), 2)
        self.assertEqual(len(ds.completed_o_clusters[0].features_array), 1)
        self.assertEqual(len(ds.completed_o_clusters[0].time_array), 1)
        self.assertEqual(len(ds.completed_o_clusters[1].features_array), 1)
        self.assertEqual(len(ds.completed_o_clusters[1].time_array), 1)

    def test_request_clustering(self):
        """
        This test is designed to check that the request_clustering works  as expected, i.e. that the DBScan creates
            the expected clusters based on p-micro-clusters' centers.
        """

        eps = 1
        lambd = 1
        beta = 0.9
        mu = 2
        min_samples = 2

        # Potential cluster 1-4.
        c1 = MicroCluster(1, lambd)
        x1_c1 = np.array([4.0, 4.0]).reshape((1, 2))
        c1.append(1, x1_c1)
        c1.update_parameters(time=1)

        c2 = MicroCluster(1, lambd)
        x1_c2 = np.array([3.9, 3.9]).reshape((1, 2))
        c2.append(1, x1_c2)
        c2.update_parameters(time=1)

        c3 = MicroCluster(1, lambd)
        x1_c3 = np.array([-4.0, -4.0]).reshape((1, 2))
        c3.append(1, x1_c3)
        c3.update_parameters(time=1)

        c4 = MicroCluster(1, lambd)
        x1_c4 = np.array([-3.9, -3.9]).reshape((1, 2))
        c4.append(1, x1_c4)
        c4.update_parameters(time=1)

        # Outlier cluster 1-2.
        c5 = MicroCluster(1, lambd)
        x1_c5 = np.array([4.0, -4.0]).reshape((1, 2))
        c5.append(1, x1_c5)
        c5.update_parameters(time=1)

        c6 = MicroCluster(1, lambd)
        x1_c6 = np.array([-3.9, 3.9]).reshape((1, 2))
        c6.append(1, x1_c6)
        c6.update_parameters(time=1)

        # Creating DenStream and appending the micro-cluster.
        ds = DenStream(eps, beta, mu, lambd, min_samples)
        ds.p_micro_clusters.append(c1)
        ds.p_micro_clusters.append(c2)
        ds.p_micro_clusters.append(c3)
        ds.p_micro_clusters.append(c4)
        ds.p_micro_clusters.append(c5)
        ds.p_micro_clusters.append(c6)

        labels = ds._request_clustering()
        expected_labels = np.array([0, 0, 1, 1, -1, -1])
        self.assertTrue(np.array_equal(labels, expected_labels))

    def test_compute_metrics(self):
        """
        This test is designed to check that the metrics computation is done correctly.
        It uses sklearn.metrics.homogeneity_score.
        """

        eps = 1
        lambd = 1
        beta = 0.9
        mu = 2
        min_samples = 2

        #  Creating potential cluster 1.
        x1_c1 = np.array([4.1, 3.9]).reshape((1, 2))
        x2_c1 = np.array([4.0, 4.0]).reshape((1, 2))

        c1 = MicroCluster(1, lambd)
        c1.append(1, x1_c1, 1)
        c1.append(1, x2_c1, 1)
        c1.update_parameters(time=1)

        #  Creating potential cluster 2.
        x1_c2 = np.array([-4.1, -3.9]).reshape((1, 2))
        x2_c2 = np.array([-4.2, -4.1]).reshape((1, 2))

        c2 = MicroCluster(1, lambd)
        c2.append(1, x1_c2, 2)
        c2.append(1, x2_c2, 2)
        c2.update_parameters(time=1)

        # Creating potential cluster 3.
        x1_c3 = np.array([4.1, 3.9]).reshape((1, 2))

        c3 = MicroCluster(1, lambd)
        c3.append(1, x1_c3, 1)
        c3.update_parameters(time=1)

        # Creating potential cluster 4.
        x1_c4 = np.array([-4.2, 4.1]).reshape((1, 2))

        c4 = MicroCluster(1, lambd)
        c4.append(1, x1_c4, 1)
        c4.update_parameters(time=1)

        # Creating DenStream and appending the micro-clusters.
        ds = DenStream(
            eps,
            beta,
            mu,
            lambd,
            min_samples,
            label_metrics_list=[metrics.homogeneity_score],
        )
        ds.p_micro_clusters.append(c1)
        ds.p_micro_clusters.append(c2)
        ds.p_micro_clusters.append(c3)
        ds.p_micro_clusters.append(c4)

        pred_labels = ds._request_clustering()
        computed_value = ds._compute_label_metrics(pred_labels)[0]["value"]
        self.assertTrue(np.abs(computed_value - 0.5) < self.TOL)

    def test_int_list_request_period(self):
        """
        This test checks that we get the same evaluation metrics for request_period with int and lists.
        """

        eps = 0.3
        lambd = 0.1
        beta = 0.2
        mu = 10
        min_samples = 1
        label_metrics_list = [metrics.homogeneity_score, metrics.completeness_score]
        no_label_metrics_list = [
            metrics.silhouette_score,
            metrics.calinski_harabasz_score,
        ]

        gen_int = generate_test_data()
        ds_int = DenStream(eps, beta, mu, lambd, min_samples, label_metrics_list, no_label_metrics_list)
        ds_int.fit_generator(gen_int, request_period=100, normalize=True)

        gen_list = generate_test_data()
        ds_list = DenStream(eps, beta, mu, lambd, min_samples, label_metrics_list, no_label_metrics_list)
        ds_list.fit_generator(gen_list, request_period=[100, 200, 300, 400], normalize=True)

        for i in range(len(ds_int.metrics_results)):
            int_metrics_i = ds_int.metrics_results[i]["metrics"]
            list_metrics_i = ds_list.metrics_results[i]["metrics"]

            for j in range(len(int_metrics_i)):
                self.assertTrue(np.abs(int_metrics_i[j]["value"] - list_metrics_i[j]["value"]) < self.TOL)

    def test_set_cluster_method(self):
        """
        This test checks that setting a new clustering method works.
        """

        eps = 0.3
        lambd = 0.1
        beta = 0.2
        mu = 10
        min_samples = 1
        label_metrics_list = [metrics.homogeneity_score, metrics.completeness_score]
        no_label_metrics_list = [
            metrics.silhouette_score,
            metrics.calinski_harabasz_score,
        ]

        gen_int = generate_test_data()
        ds_int = DenStream(eps, beta, mu, lambd, min_samples, label_metrics_list, no_label_metrics_list)
        model_int = KMeans(n_clusters=2, random_state=42)
        ds_int.set_clustering_model(model_int)
        ds_int.fit_generator(gen_int, request_period=100, normalize=True)

        gen_list = generate_test_data()
        ds_list = DenStream(eps, beta, mu, lambd, min_samples, label_metrics_list, no_label_metrics_list)
        model_list = KMeans(n_clusters=2, random_state=42)
        ds_list.set_clustering_model(model_list)
        ds_list.fit_generator(gen_list, request_period=[100, 200, 300, 400], normalize=True)

        for i in range(len(ds_int.metrics_results)):
            int_metrics_i = ds_int.metrics_results[i]["metrics"]
            list_metrics_i = ds_list.metrics_results[i]["metrics"]

            for j in range(len(int_metrics_i)):
                self.assertTrue(np.abs(int_metrics_i[j]["value"] - list_metrics_i[j]["value"]) < self.TOL)


if __name__ == "__main__":
    unittest.main()
