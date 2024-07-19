import numpy as np


def do_Factor_VAE(generated_latent, label, n_train, batch_size, random_state, verbose=False):
    global_variances = np.var(generated_latent[:n_train], axis=0, ddof=1)
    print("global_variances\t", global_variances.shape)
    active_dims = _prune_dims(global_variances)
    print("active_dims: ", active_dims)

    score = {}
    if not active_dims.any():
        score["train_accuracy"] = 0.
        score["eval_accuracy"] = 0.
        score["num_active_dims"] = 0
        return score

    training_votes = generate_samples(generated_latent[:n_train], label[:n_train], global_variances, batch_size, active_dims)
    eval_votes = generate_samples(generated_latent[n_train:], label[n_train:], global_variances, batch_size, active_dims)

    classifier = np.argmax(training_votes, axis=0)
    other_index = np.arange(training_votes.shape[1])
    train_accuracy = np.sum(training_votes[classifier, other_index]) * 1. / np.sum(training_votes)
    eval_accuracy = np.sum(eval_votes[classifier, other_index]) * 1. / np.sum(eval_votes)

    score = {}
    score["train_accuracy"] = train_accuracy
    score["eval_accuracy"] = eval_accuracy
    score["num_active_dims"] = len(active_dims)
    return score


def _prune_dims(variances, threshold=0.):
    """Mask for dimensions collapsed to the prior."""
    scale_z = np.sqrt(variances)
    return scale_z >= threshold


def sample_representations_with_target_factor(z, y, factor_index, factor_value, batch_size):
    n_point = z.shape[0]

    repr_list, factor_list = [], []
    count = 0

    idx_list = np.arange(n_point)
    np.random.shuffle(idx_list)

    for idx in idx_list:
        repr_ = z[idx]
        factor_ = y[idx]
        if factor_[factor_index] != factor_value:
            continue
        repr_list.append(repr_)
        factor_list.append(factor_)
        count += 1
        if count == batch_size:
            break
    
    repr_list = np.array(repr_list)
    factor_list = np.array(factor_list)
    return repr_list, factor_list


def generate_samples(z, y, global_variances, batch_size, active_dims):
    n_point, n_factor = y.shape
    votes = np.zeros((n_factor, global_variances.shape[0]), dtype=np.int64)

    for idx in range(n_point):
        factor_index = np.random.randint(n_factor)
        repr_list, factor_list = sample_representations_with_target_factor(
            z=z, y=y, factor_index=factor_index, factor_value=y[idx, factor_index], batch_size=batch_size)
        local_variances = np.var(repr_list, axis=0, ddof=1)
        argmin = np.argmin(local_variances[active_dims] / global_variances[active_dims])

        votes[factor_index, argmin] += 1
    print("votes: ", votes.shape)
    return votes
