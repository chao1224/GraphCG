import numpy as np
import sklearn


def do_MIG(generated_latent, label, n_train):
    z_train, y_train = generated_latent[:n_train].T, label[:n_train].T
    z_val, y_val = generated_latent[n_train:].T, label[n_train:].T

    score = {}
    score['train'] = _compute_mig(z_train, y_train)
    score['val'] = _compute_mig(z_val, y_val)
    return score


def _compute_mig(mus_train, ys_train):
    """Computes score based on both training and testing codes and factors."""
    score_dict = {}
    discretized_mus = make_discretizer(mus_train)
    m = discrete_mutual_info(discretized_mus, ys_train)
    assert m.shape[0] == mus_train.shape[0]
    assert m.shape[1] == ys_train.shape[0]
    # m is [num_latents, num_factors]
    entropy = discrete_entropy(ys_train)
    sorted_m = np.sort(m, axis=0)[::-1]
    score_dict["discrete_mig"] = np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
    return score_dict


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


def discrete_entropy(ys):
    """Compute discrete mutual information."""
    num_factors = ys.shape[0]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
    return h
