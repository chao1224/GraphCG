import numpy as np
from sklearn import linear_model


def do_Beta_VAE_single_factor(generated_latent, label, n_train, random_state, batch_size, verbose=False):
    # for pre-processing
    z_train, y_train = generated_latent[:n_train], label[:n_train]
    z_val, y_val = generated_latent[n_train:], label[n_train:]
    if verbose:
        print("z_train: {}\ty_train: {}.".format(z_train.shape, y_train.shape))
        print("z_val: {}\ty_val: {}.".format(z_val.shape, y_val.shape))

    model = linear_model.LogisticRegression(random_state=random_state, solver='liblinear')
    model.fit(z_train, y_train)

    train_accuracy = model.score(z_train, y_train)
    train_accuracy = np.mean(model.predict(z_train) == y_train)

    eval_accuracy = model.score(z_val, y_val)

    score = {}
    score["train_accuracy"] = train_accuracy
    score["eval_accuracy"] = eval_accuracy
    return score


def do_Beta_VAE(generated_latent, label, n_train, random_state, batch_size, verbose=False):
    # for pre-processing
    z_train, y_train = generate_samples(generated_latent[:n_train], label[:n_train], batch_size)
    z_val, y_val = generate_samples(generated_latent[n_train:], label[n_train:], batch_size)
    if verbose:
        print("z_train: {}\ty_train: {}.".format(z_train.shape, y_train.shape))
        print("z_val: {}\ty_val: {}.".format(z_val.shape, y_val.shape))

    model = linear_model.LogisticRegression(random_state=random_state, solver='liblinear')
    model.fit(z_train, y_train)

    train_accuracy = model.score(z_train, y_train)
    train_accuracy = np.mean(model.predict(z_train) == y_train)

    eval_accuracy = model.score(z_val, y_val)

    score = {}
    score["train_accuracy"] = train_accuracy
    score["eval_accuracy"] = eval_accuracy
    return score


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


def generate_samples(z, y, batch_size):
    n_point, n_factor = y.shape

    neo_z, neo_y = [], []
    for idx in range(n_point):
        for idx_factor in range(n_factor):
            repr_list, _ = sample_representations_with_target_factor(
                z=z, y=y, factor_index=idx_factor, factor_value=y[idx][idx_factor], batch_size=batch_size)
            
            feature = np.mean(np.abs(repr_list - z[idx]), axis=0)
            neo_z.append(feature)
            neo_y.append(idx_factor)

    neo_z = np.array(neo_z)
    neo_y = np.array(neo_y)
    neo_y = np.expand_dims(neo_y, axis=1)
    return neo_z, neo_y
