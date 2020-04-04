import numpy as np
from sklearn import metrics

from pyDenStream.den_stream import DenStream
from pyDenStream.micro_cluster import MicroCluster


TOL = 1e-6


def test_closest_cluster():
    """
    This test is designed to check that we find the expected closest cluster (i.e. index) for a given data point.
    """

    eps = 1
    lambd = 1
    beta = 0.5
    mu = 3
    min_samples = 3

    # Creating potential cluster 1.
    c1 = MicroCluster(1, lambd)
    x1_c1 = np.array([3.9, 3.9]).reshape((1, 2))
    c1.append(1, x1_c1)
    x2_c1 = np.array([3.9, 3.9]).reshape((1, 2))
    c1.append(1, x2_c1)
    c1.update_parameters(time=1)

    # Creating potential cluster 2.
    c2 = MicroCluster(1, lambd)
    x1_c2 = np.array([4.0, 4.0]).reshape((1, 2))
    c2.append(1, x1_c2)
    x2_c2 = np.array([4.0, 4.0]).reshape((1, 2))
    c2.append(1, x2_c2)
    c2.update_parameters(time=1)

    # Creating outlier cluster 1.
    c3 = MicroCluster(1, lambd)
    x1_c3 = np.array([0.0, 0.0]).reshape((1, 2))
    c3.append(1, x1_c3)
    x2_c3 = np.array([0.0, 0.0]).reshape((1, 2))
    c3.append(1, x2_c3)
    c3.update_parameters(time=1)

    # DenStream setup.
    ds = DenStream(eps, beta, mu, lambd, min_samples)
    ds.p_micro_clusters.append(c1)
    ds.p_micro_clusters.append(c2)
    ds.o_micro_clusters.append(c3)

    feature_array = np.array([4.1, 4.1]).reshape((1, 2))
    closest_index = ds._find_closest_cluster(ds.p_micro_clusters, feature_array)
    assert(closest_index == 1)


def test_merging_points():
    """
    This test is designed to check that merging a data point into a micro-cluster works as expected.
    The merging is done for two points, one should go to the p-micro-cluster indexed 0 and the other one should
        go to the o-micro-cluster indexed 0.
    """

    eps = 1
    lambd = 1
    beta = 0.5
    mu = 3
    min_samples = 3

    # Creating potential cluster 1.
    c1 = MicroCluster(1, lambd)
    x1_c1 = np.array([4.1, 3.9]).reshape((1, 2))
    c1.append(1, x1_c1)
    x2_c1 = np.array([4.0, 4.0]).reshape((1, 2))
    c1.append(1, x2_c1)
    c1.update_parameters(time=1)

    # Creating potential cluster 2.
    c2 = MicroCluster(1, lambd)
    x1_c2 = np.array([-4.1, -3.9]).reshape((1, 2))
    c2.append(1, x1_c2)
    x2_c2 = np.array([-4.2, -4.1]).reshape((1, 2))
    c2.append(1, x2_c2)
    c2.update_parameters(time=1)

    # Creating outlier cluster 1.
    c3 = MicroCluster(1, lambd)
    x1_c3 = np.array([0.0, 0.0]).reshape((1, 2))
    c3.append(1, x1_c3)
    x2_c3 = np.array([0.0, 0.0]).reshape((1, 2))
    c3.append(1, x2_c3)
    c3.update_parameters(time=1)

    # DenStream setup.
    ds = DenStream(eps, beta, mu, lambd, min_samples)
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
    assert(len(ds.p_micro_clusters) == 2)
    assert(len(ds.p_micro_clusters[0].features_array) == 3)
    assert(len(ds.p_micro_clusters[1].features_array) == 2)

    assert(len(ds.o_micro_clusters) == 2)
    assert(len(ds.o_micro_clusters[0].features_array) == 2)
    assert(len(ds.o_micro_clusters[1].features_array) == 1)


def test_no_clusters():
    """
    This test is designed to check that merging a point when we have no p-micro-clusters and no o-micro-clusters
        works as expected.
    The expected outcome is that it will create a new o-micro-cluster.
    """

    eps = 1
    lambd = 1
    beta = 1.0
    mu = 3
    min_samples = 3

    ds = DenStream(eps, beta, mu, lambd, min_samples)
    assert(len(ds.p_micro_clusters) == 0)
    assert (len(ds.o_micro_clusters) == 0)

    feature_array_1 = np.array([100.0, 100.0]).reshape((1, 2))
    ds._merging(1, feature_array_1)

    assert(len(ds.p_micro_clusters) == 0)
    assert (len(ds.o_micro_clusters) == 1)

    feature_array_2 = np.array([-100.0, -100.0]).reshape((1, 2))
    ds._merging(1, feature_array_2)

    assert (len(ds.p_micro_clusters) == 0)
    assert (len(ds.o_micro_clusters) == 2)


def test_moving_o_to_p_cluster():
    """
    This test is designed to check that moving a cluster from the outlier to a potential cluster, if w < beta * mu.
    """

    eps = 1
    lambd = 1
    beta = 0.5
    mu = 3
    min_samples = 3

    # Creating potential cluster 1.
    c1 = MicroCluster(1, lambd)
    x1_c1 = np.array([4.1, 3.9]).reshape((1, 2))
    c1.append(1, x1_c1)
    x2_c1 = np.array([4.0, 4.0]).reshape((1, 2))
    c1.append(1, x2_c1)
    c1.update_parameters(time=1)

    # Creating potential cluster 2.
    c2 = MicroCluster(1, lambd)
    x1_c2 = np.array([0.0, 0.0]).reshape((1, 2))
    c2.append(1, x1_c2)
    x2_c2 = np.array([0.0, 0.0]).reshape((1, 2))
    c2.append(1, x2_c2)
    c2.update_parameters(time=1)

    # Creating outlier cluster 1.
    c3 = MicroCluster(1, lambd)
    x1_c3 = np.array([-4.0, -4.0]).reshape((1, 2))
    c3.append(1, x1_c3)
    x2_c3 = np.array([-4.0, -4.1]).reshape((1, 2))
    c3.append(1, x2_c3)
    c3.update_parameters(time=1)

    # DenStream setup.
    ds = DenStream(eps, beta, mu, lambd, min_samples)
    ds.p_micro_clusters.append(c1)
    ds.o_micro_clusters.append(c2)
    ds.o_micro_clusters.append(c3)

    # Defining test data point.
    o_potential = np.array([0.1, 0.1]).reshape((1, 2))
    ds._merging(1, o_potential)

    # Testing that the p and o micro-clusters has the expected number of clusters.
    assert(len(ds.o_micro_clusters) == 1)
    assert(len(ds.p_micro_clusters) == 2)

    # Testing that p and o clusters contains the expected number of points.
    assert(len(ds.p_micro_clusters[0].time_array) == 2)
    assert (len(ds.p_micro_clusters[0].features_array) == 2)
    assert(len(ds.p_micro_clusters[1].time_array) == 3)
    assert (len(ds.p_micro_clusters[1].features_array) == 3)

    assert(len(ds.o_micro_clusters[0].time_array) == 2)
    assert (len(ds.o_micro_clusters[0].features_array) == 2)


def test_fit_generator_cluster():
    """
    This test is designed to check that the expected number of p/o-micro-clusters are formed from a stream (generator).
    The input have two expected p-micro-clusters and three expected o-micro-clusters.
    """

    x_inputs = np.array([
        [4.0, 4.0], [-4.0, -4.0], [3.99, 3.99], [-10.0, -10.0], [4.01, 4.01],
        [-4.01, -4.01], [300.0, 300.0], [10.0, -10.0]
    ])

    time_input = [1, 1, 1, 1, 1, 1, 1, 1]

    def generator(feature_arrays, time_list):
        for i in range(0, len(time_input)):
            yield {
                "time": time_list[i],
                "feature_array": feature_arrays[i, :].reshape((1, 2))
            }

    eps = 1
    lambd = 1
    beta = 0.5
    mu = 3
    min_samples = 3

    ds = DenStream(eps, beta, mu, lambd, min_samples)
    gen = generator(x_inputs, time_input)

    ds.fit_generator(gen)

    assert(len(ds.o_micro_clusters) == 3)
    assert(len(ds.p_micro_clusters) == 2)
    assert(len(ds.completed_o_clusters) == 0)
    assert(len(ds.completed_p_clusters) == 0)


def test_fit_generator_fading():
    """
    This test is designed to check that the micro-clusters are fading, i.e. the activate micro-clusters are moved
        to the completed ones.
    """

    x_inputs = np.array([
        [-4.0, -4.0], [4.0, 4.0], [3.99, 3.99], [-10.0, 10.0], [4.01, 4.01],
        [-4.01, -4.01], [300.0, 300.0], [10.0, -10.0]
    ])


    time_input = [4, 1, 1, 1, 1,
                  4, 1, 4]

    def generator(feature_arrays, time_list):
        for i in range(0, len(time_input)):
            print(i)
            yield {
                "time": time_list[i],
                "feature_array": feature_arrays[i, :].reshape((1, 2))
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
    assert(len(ds.o_micro_clusters) == 1)
    assert(len(ds.o_micro_clusters[0].features_array) == 1)
    assert(len(ds.o_micro_clusters[0].time_array) == 1)

    assert(len(ds.p_micro_clusters) == 1)
    assert(len(ds.p_micro_clusters[0].features_array) == 2)
    assert(len(ds.p_micro_clusters[0].time_array) == 2)

    # Checking that the inactivate p/o-micro-clusters have the expected size / number of data-points.
    assert(len(ds.completed_p_clusters) == 1)
    assert(len(ds.completed_p_clusters[0].features_array) == 3)
    assert(len(ds.completed_p_clusters[0].time_array) == 3)

    assert(len(ds.completed_o_clusters) == 2)
    assert(len(ds.completed_o_clusters[0].features_array) == 1)
    assert(len(ds.completed_o_clusters[0].time_array) == 1)
    assert(len(ds.completed_o_clusters[1].features_array) == 1)
    assert(len(ds.completed_o_clusters[1].time_array) == 1)


def test_request_clustering():
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
    assert(np.array_equal(labels, expected_labels))


def test_compute_metrics():
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

    c1  = MicroCluster(1, lambd)
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
    ds = DenStream(eps, beta, mu, lambd, min_samples,  label_metrics_list=[metrics.homogeneity_score])
    ds.p_micro_clusters.append(c1)
    ds.p_micro_clusters.append(c2)
    ds.p_micro_clusters.append(c3)
    ds.p_micro_clusters.append(c4)

    pred_labels = ds._request_clustering()
    computed_value = ds._compute_label_metrics(pred_labels)[0]["value"]
    assert(np.abs(computed_value - 0.5) < TOL)


def test_int_list_request_period():
    """
    This test checks that we get the same evaluation metrics for request_period with int and lists.
    """

    np.random.seed(42)

    num_samples = 100
    num_features = 2

    sigma = 0.1

    # Generating data for cluster 1.
    center_1 = np.array([1.0, 1.0]).reshape((1, num_features))
    x_1 = center_1 + np.random.normal(0.0, sigma, [num_samples, num_features])
    y_1 = np.repeat(0, num_samples).reshape((num_samples, 1))
    t_1 = np.linspace(1, 100, num=num_samples).reshape((num_samples, 1))

    # Generating data for cluster 2.
    center_2 = np.array([1.0, -1.0]).reshape((1, num_features))
    x_2 = center_2 + np.random.normal(0.0, sigma, [num_samples, num_features])
    y_2 = np.repeat(0, num_samples).reshape((num_samples, 1))
    t_2 = np.linspace(101, 200, num=num_samples).reshape((num_samples, 1))

    # Generating data for cluster 3.
    center_3 = np.array([-1.0, -1.0]).reshape((1, num_features))
    x_3 = center_3 + np.random.normal(0.0, sigma, [num_samples, num_features])
    y_3 = np.repeat(0, num_samples).reshape((num_samples, 1))
    t_3 = np.linspace(51, 150, num=num_samples).reshape((num_samples, 1))

    # Generating data for cluster 4.
    center_4 = np.array([-1.0, 1.0]).reshape((1, num_features))
    x_4 = center_4 + np.random.normal(0.0, sigma, [num_samples, num_features])
    y_4 = np.repeat(0, num_samples).reshape((num_samples, 1))
    t_4 = np.linspace(51, 150, num=num_samples).reshape((num_samples, 1))

    X = np.concatenate([x_1, x_2, x_3, x_4], axis=0).astype(float)
    Y = np.concatenate([y_1, y_2, y_3, y_4], axis=0).astype(int)
    T = np.concatenate([t_1, t_2, t_3, t_4], axis=0).astype(int)

    # Sorting data s.t. they come in time order.
    idx = np.argsort(T, axis=0).reshape(T.shape[0],)
    X = X[idx, :]
    Y = Y[idx, :]
    T = T[idx, :]

    def generator(X, Y, T):
        for i in range(0, X.shape[0]):
            yield {
                "time": int(T[i, :]),
                "feature_array": X[i, :].reshape((1, X.shape[1])),
                "label": int(Y[i, :])
            }

    eps = 0.3
    lambd = 0.1
    beta = 0.2
    mu = 10
    min_samples = 1
    label_metrics_list = [metrics.homogeneity_score, metrics.completeness_score]
    unlabel_metrics_list = [metrics.silhouette_score, metrics.calinski_harabasz_score]

    gen_int = generator(X, Y, T)
    ds_int = DenStream(eps, beta, mu, lambd, min_samples, label_metrics_list, unlabel_metrics_list)
    ds_int.fit_generator(gen_int, request_period=100, normalize=True)

    gen_list = generator(X, Y, T)
    ds_list = DenStream(eps, beta, mu, lambd, min_samples, label_metrics_list, unlabel_metrics_list)
    ds_list.fit_generator(gen_list, request_period=[100, 200, 300, 400], normalize=True)

    for i in range(len(ds_int.metrics_results)):
        int_metrics_i = ds_int.metrics_results[i]["metrics"]
        list_metrics_i = ds_list.metrics_results[i]["metrics"]

        for j in range(len(int_metrics_i)):
            assert(np.abs(int_metrics_i[j]["value"] - list_metrics_i[j]["value"]) < TOL)
