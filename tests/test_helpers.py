import numpy as np


def generator(X, Y, T):
    """
    Creating generator used for fit_generator.
    """

    for i in range(0, X.shape[0]):
        yield {
            "time": int(T[i, :]),
            "feature_array": X[i, :].reshape((1, X.shape[1])),
            "label": int(Y[i, :]),
        }


def generate_test_data():
    """
    Generating test-data.
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
    y_2 = np.repeat(1, num_samples).reshape((num_samples, 1))
    t_2 = np.linspace(101, 200, num=num_samples).reshape((num_samples, 1))

    # Generating data for cluster 3.
    center_3 = np.array([-1.0, -1.0]).reshape((1, num_features))
    x_3 = center_3 + np.random.normal(0.0, sigma, [num_samples, num_features])
    y_3 = np.repeat(2, num_samples).reshape((num_samples, 1))
    t_3 = np.linspace(51, 150, num=num_samples).reshape((num_samples, 1))

    # Generating data for cluster 4.
    center_4 = np.array([-1.0, 1.0]).reshape((1, num_features))
    x_4 = center_4 + np.random.normal(0.0, sigma, [num_samples, num_features])
    y_4 = np.repeat(3, num_samples).reshape((num_samples, 1))
    t_4 = np.linspace(51, 150, num=num_samples).reshape((num_samples, 1))

    X = np.concatenate([x_1, x_2, x_3, x_4], axis=0).astype(float)
    Y = np.concatenate([y_1, y_2, y_3, y_4], axis=0).astype(int)
    T = np.concatenate([t_1, t_2, t_3, t_4], axis=0).astype(int)

    # Sorting data s.t. they come in time order.
    idx = np.argsort(T, axis=0).reshape(
        T.shape[0],
    )
    X = X[idx, :]
    Y = Y[idx, :]
    T = T[idx, :]

    return generator(X, Y, T)
