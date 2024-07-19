import numpy as np
import sklearn
from sklearn import linear_model, metrics, preprocessing


def do_Modularity(generated_latent, label, n_train):
    z_train, y_train = generated_latent[:n_train].T, label[:n_train].T
    z_val, y_val = generated_latent[n_train:].T, label[n_train:].T

    discretized_z = make_discretizer(z_train)
    mutual_information = discrete_mutual_info(discretized_z, y_train)
    # Mutual information should have shape [num_codes, num_factors].
    assert mutual_information.shape[0] == z_train.shape[0]
    assert mutual_information.shape[1] == y_train.shape[0]

    score = {}
    score["modularity_score"] = modularity(mutual_information)

    explicitness_score_train = np.zeros([y_train.shape[0], 1])
    explicitness_score_val = np.zeros([y_val.shape[0], 1])
    z_train_norm, mean_mus, stddev_mus = normalize_data(z_train)
    z_test_norm, _, _ = normalize_data(z_val, mean_mus, stddev_mus)
    for i in range(y_train.shape[0]):
        explicitness_score_train[i], explicitness_score_val[i] = explicitness_per_factor(z_train_norm, y_train[i, :], z_test_norm, y_val[i, :])
    score["explicitness_score_train"] = np.mean(explicitness_score_train)
    score["explicitness_score_val"] = np.mean(explicitness_score_val)
    return score


def histogram_discretize(target, num_bins):
    """Discretization based on histograms."""
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(target[i, :], num_bins)[1][:-1])
    return discretized


def make_discretizer(target, num_bins=20, discretizer_fn=histogram_discretize):
    """Wrapper that creates discretizers."""
    return discretizer_fn(target, num_bins)


def discrete_mutual_info(mus, ys):
    """Compute discrete mutual information."""
    num_codes = mus.shape[0]
    num_factors = ys.shape[0]
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :], mus[i, :])
    return m


def normalize_data(data, mean=None, stddev=None):
    if mean is None:
        mean = np.mean(data, axis=1)
    if stddev is None:
        stddev = np.std(data, axis=1)
    return (data - mean[:, np.newaxis]) / stddev[:, np.newaxis], mean, stddev


def explicitness_per_factor(z_train, y_train, z_test, y_test):
    """Compute explicitness score for a factor as ROC-AUC of a classifier.
    Args:
    z_train: Representation for training, (num_codes, num_points)-np array.
    y_train: Ground truth factors for training, (num_factors, num_points)-np
        array.
    z_test: Representation for testing, (num_codes, num_points)-np array.
    y_test: Ground truth factors for testing, (num_factors, num_points)-np
        array.
    Returns:
    roc_train: ROC-AUC score of the classifier on training data.
    roc_test: ROC-AUC score of the classifier on testing data.
    """
    x_train = np.transpose(z_train)
    x_test = np.transpose(z_test)
    clf = linear_model.LogisticRegression().fit(x_train, y_train)
    y_pred_train = clf.predict_proba(x_train)
    y_pred_test = clf.predict_proba(x_test)
    mlb = preprocessing.MultiLabelBinarizer()
    y_train = np.expand_dims(y_train, 1)
    y_test = np.expand_dims(y_test, 1)
    roc_train = metrics.roc_auc_score(mlb.fit_transform(y_train), y_pred_train)
    roc_test = metrics.roc_auc_score(mlb.fit_transform(y_test), y_pred_test)
    return roc_train, roc_test


def modularity(mutual_information):
    """Computes the modularity from mutual information."""
    # Mutual information has shape [num_codes, num_factors].
    squared_mi = np.square(mutual_information)
    max_squared_mi = np.max(squared_mi, axis=1)
    numerator = np.sum(squared_mi, axis=1) - max_squared_mi
    denominator = max_squared_mi * (squared_mi.shape[1] -1.)
    delta = numerator / denominator
    modularity_score = 1. - delta
    index = (max_squared_mi == 0.)
    modularity_score[index] = 0.
    return np.mean(modularity_score)