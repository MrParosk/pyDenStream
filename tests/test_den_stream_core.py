import unittest

import numpy as np

from denstream.den_stream import DenStream
from denstream.micro_cluster import MicroCluster


class TestDenStreamCore(unittest.TestCase):
    def setUp(self):
        self.TOL = 1e-6
        self.eps = 1
        self.lambd = 1
        self.beta = 0.5
        self.mu = 3
        self.min_samples = 3

    def test_closest_cluster(self):
        """
        This test is designed to check that we find the expected closest cluster (i.e. index) for a given data point.
        """

        # Creating potential cluster 1.
        c1 = MicroCluster(1, self.lambd)
        x1_c1 = np.array([3.9, 3.9]).reshape((1, 2))
        c1.append(1, x1_c1)
        x2_c1 = np.array([3.9, 3.9]).reshape((1, 2))
        c1.append(1, x2_c1)
        c1.update_parameters(time=1)

        # Creating potential cluster 2.
        c2 = MicroCluster(1, self.lambd)
        x1_c2 = np.array([4.0, 4.0]).reshape((1, 2))
        c2.append(1, x1_c2)
        x2_c2 = np.array([4.0, 4.0]).reshape((1, 2))
        c2.append(1, x2_c2)
        c2.update_parameters(time=1)

        # Creating outlier cluster 1.
        c3 = MicroCluster(1, self.lambd)
        x1_c3 = np.array([0.0, 0.0]).reshape((1, 2))
        c3.append(1, x1_c3)
        x2_c3 = np.array([0.0, 0.0]).reshape((1, 2))
        c3.append(1, x2_c3)
        c3.update_parameters(time=1)

        # DenStream setup.
        ds = DenStream(self.eps, self.beta, self.mu, self.lambd, self.min_samples)
        ds.p_micro_clusters.append(c1)
        ds.p_micro_clusters.append(c2)
        ds.o_micro_clusters.append(c3)

        feature_array = np.array([4.1, 4.1]).reshape((1, 2))
        closest_index = ds._find_closest_cluster(ds.p_micro_clusters, feature_array)
        self.assertTrue(closest_index == 1)

    def test_merging_points(self):
        """
        This test is designed to check that merging a data point into a micro-cluster works as expected.
        The merging is done for two points, one should go to the p-micro-cluster indexed 0 and the other one should
            go to the o-micro-cluster indexed 0.
        """

        # Creating potential cluster 1.
        c1 = MicroCluster(1, self.lambd)
        x1_c1 = np.array([4.1, 3.9]).reshape((1, 2))
        c1.append(1, x1_c1)
        x2_c1 = np.array([4.0, 4.0]).reshape((1, 2))
        c1.append(1, x2_c1)
        c1.update_parameters(time=1)

        # Creating potential cluster 2.
        c2 = MicroCluster(1, self.lambd)
        x1_c2 = np.array([-4.1, -3.9]).reshape((1, 2))
        c2.append(1, x1_c2)
        x2_c2 = np.array([-4.2, -4.1]).reshape((1, 2))
        c2.append(1, x2_c2)
        c2.update_parameters(time=1)

        # Creating outlier cluster 1.
        c3 = MicroCluster(1, self.lambd)
        x1_c3 = np.array([0.0, 0.0]).reshape((1, 2))
        c3.append(1, x1_c3)
        x2_c3 = np.array([0.0, 0.0]).reshape((1, 2))
        c3.append(1, x2_c3)
        c3.update_parameters(time=1)

        # DenStream setup.
        ds = DenStream(self.eps, self.beta, self.mu, self.lambd, self.min_samples)
        ds.p_micro_clusters.append(c1)
        ds.p_micro_clusters.append(c2)
        ds.o_micro_clusters.append(c3)

        # Defining test data points.
        p_potential = np.array([4.1, 3.9]).reshape((1, 2))
        ds._merging(1, p_potential)

        p_outlier = np.array([7.0, 7.0]).reshape((1, 2))
        ds._merging(1, p_outlier)

        # Testing that the data point get to the correct cluster,
        # i.e. p_potential to c1 and p_outlier to a new outlier cluster.
        self.assertEqual(len(ds.p_micro_clusters), 2)
        self.assertEqual(len(ds.p_micro_clusters[0].features_array), 3)
        self.assertEqual(len(ds.p_micro_clusters[1].features_array), 2)

        self.assertEqual(len(ds.o_micro_clusters), 2)
        self.assertEqual(len(ds.o_micro_clusters[0].features_array), 2)
        self.assertEqual(len(ds.o_micro_clusters[1].features_array), 1)

    def test_no_clusters(self):
        """
        This test is designed to check that merging a point when we have no p-micro-clusters and no o-micro-clusters
            works as expected.
        The expected outcome is that it will create a new o-micro-cluster.
        """

        beta = 1.0

        ds = DenStream(self.eps, beta, self.mu, self.lambd, self.min_samples)
        self.assertEqual(len(ds.p_micro_clusters), 0)
        self.assertEqual(len(ds.o_micro_clusters), 0)

        feature_array_1 = np.array([100.0, 100.0]).reshape((1, 2))
        ds._merging(1, feature_array_1)

        self.assertEqual(len(ds.p_micro_clusters), 0)
        self.assertEqual(len(ds.o_micro_clusters), 1)

        feature_array_2 = np.array([-100.0, -100.0]).reshape((1, 2))
        ds._merging(1, feature_array_2)

        self.assertEqual(len(ds.p_micro_clusters), 0)
        self.assertEqual(len(ds.o_micro_clusters), 2)

    def test_moving_o_to_p_cluster(self):
        """
        This test is designed to check that moving a cluster from the outlier to a potential cluster, if w < beta * mu.
        """

        # Creating potential cluster 1.
        c1 = MicroCluster(1, self.lambd)
        x1_c1 = np.array([4.1, 3.9]).reshape((1, 2))
        c1.append(1, x1_c1)
        x2_c1 = np.array([4.0, 4.0]).reshape((1, 2))
        c1.append(1, x2_c1)
        c1.update_parameters(time=1)

        # Creating potential cluster 2.
        c2 = MicroCluster(1, self.lambd)
        x1_c2 = np.array([0.0, 0.0]).reshape((1, 2))
        c2.append(1, x1_c2)
        x2_c2 = np.array([0.0, 0.0]).reshape((1, 2))
        c2.append(1, x2_c2)
        c2.update_parameters(time=1)

        # Creating outlier cluster 1.
        c3 = MicroCluster(1, self.lambd)
        x1_c3 = np.array([-4.0, -4.0]).reshape((1, 2))
        c3.append(1, x1_c3)
        x2_c3 = np.array([-4.0, -4.1]).reshape((1, 2))
        c3.append(1, x2_c3)
        c3.update_parameters(time=1)

        # DenStream setup.
        ds = DenStream(self.eps, self.beta, self.mu, self.lambd, self.min_samples)
        ds.p_micro_clusters.append(c1)
        ds.o_micro_clusters.append(c2)
        ds.o_micro_clusters.append(c3)

        # Defining test data point.
        o_potential = np.array([0.1, 0.1]).reshape((1, 2))
        ds._merging(1, o_potential)

        # Testing that the p and o micro-clusters has the expected number of clusters.
        self.assertEqual(len(ds.o_micro_clusters), 1)
        self.assertEqual(len(ds.p_micro_clusters), 2)

        # Testing that p and o clusters contains the expected number of points.
        self.assertEqual(len(ds.p_micro_clusters[0].time_array), 2)
        self.assertEqual(len(ds.p_micro_clusters[0].features_array), 2)
        self.assertEqual(len(ds.p_micro_clusters[1].time_array), 3)
        self.assertEqual(len(ds.p_micro_clusters[1].features_array), 3)

        self.assertEqual(len(ds.o_micro_clusters[0].time_array), 2)
        self.assertEqual(len(ds.o_micro_clusters[0].features_array), 2)


if __name__ == "__main__":
    unittest.main()
