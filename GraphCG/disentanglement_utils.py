import os

import numpy as np

from .metrics_beta_vae import do_Beta_VAE, do_Beta_VAE_single_factor
from .metrics_dci import do_DCI
from .metrics_factor_vae import do_Factor_VAE
from .metrics_mig import do_MIG
from .metrics_modularity import do_Modularity
from .metrics_sap import do_SAP

EPS = 1e-12


def normalize(X, mean=None, stddev=None, useful_features=None, remove_constant=True):
    calc_mean, calc_stddev = False, False
    
    if mean is None:
        mean = np.mean(X, 0) # training set
        calc_mean = True
    
    if stddev is None:
        stddev = np.std(X, 0) # training set
        calc_stddev = True
        useful_features = np.nonzero(stddev)[0] # inconstant features, ([0]=shape correction)
    
    if remove_constant and useful_features is not None:
        X = X[:, useful_features]
        if calc_mean:
            mean = mean[useful_features]
        if calc_stddev:
            stddev = stddev[useful_features]
    
    X_zm = X - mean    
    X_zm_unit = X_zm / stddev
    
    return X_zm_unit, mean, stddev, useful_features


def norm_entropy(p):
    '''p: probabilities '''
    n = p.shape[0]
    return - p.dot(np.log(p + EPS) / np.log(n + EPS))


def entropic_scores(r):
    '''r: relative importances '''
    r = np.abs(r)
    ps = r / np.sum(r, axis=0) # 'probabilities'
    # ps = np.where(r!=0,r/np.sum(r,axis=0),0)
    hs = [1-norm_entropy(p) for p in ps.T]
    return hs


def mse(predicted, target):
    ''' mean square error '''
    predicted = predicted[:, None] if len(predicted.shape) == 1 else predicted #(n,)->(n,1)
    target = target[:, None] if len(target.shape) == 1 else target #(n,)->(n,1)
    err = predicted - target
    err = err.T.dot(err) / len(err)
    return err[0, 0] #value not array


def rmse(predicted, target):
    ''' root mean square error '''
    return np.sqrt(mse(predicted, target))


def nmse(predicted, target):
    ''' normalized mean square error '''
    return mse(predicted, target) / np.var(target)


def nrmse(predicted, target):
    ''' normalized root mean square error '''
    return rmse(predicted, target) / np.std(target)


def print_table_pretty(name, values, factor_label, model_names):
    headers = [factor_label + str(i) for i in range(len(values[0]))]
    headers[-1] = "Avg."
    headers = "\t" + "\t".join(headers)
    print("{0}:\n{1}".format(name, headers))
    
    for i, values in enumerate(values):
        value = ""
        for v in values:
            value +=  "{0:.2f}".format(v) + "&\t"
        print("{0}\t{1}".format(model_names[i], value))
    print("") #newline
