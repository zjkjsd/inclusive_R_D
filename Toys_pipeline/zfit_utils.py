# +
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tool to apply offline cuts (and DecayHash) to Ntuples and save/split them to one/multiple parquet/root files.

Usage: python3 1_Apply_DecayHash.py -d folder -i -o -l -n (--nohash)

Example: python3 1_Apply_DecayHash.py -d Samples/Signal_MC14ri/MC14ri_sigDDst_bengal_e_2 \
         -i sigDDst_bengal_e_2.root -o parquet -l e --mctype signal (--nohash)
"""

import numpy as np

# import tensorflow as tf
import uncertainties
import zfit
import zfit.z.numpy as znp
from uncertainties import ufloat

import uuid


def zfit_gauss(x):
    """
        Calculates a simple gaussian fit given an array x.
        Returns mu, sigma as correlated ufloats.
        
        Intended for fitting gaussian to toyMC results.
    """
    # get starting values and ranges:
    mean = np.mean(x)
    std = np.std(x)

    # zfit treats the name as a global variable
    # if we want to loop several fits need to create a unique name for each parameter
    unique_id = str(uuid.uuid4())

    # create the model
    obs = zfit.Space('x', limits=(-10, 10))
    mu = zfit.Parameter('mu_' + unique_id, mean, mean - 2 * std, mean + 2 * std)
    sigma = zfit.Parameter('sigma_' + unique_id, std, 0.1 * std, 10 * std)
    params = [mu, sigma]
    gauss = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)

    data = zfit.Data.from_numpy(obs=obs, array=np.array(x))

    nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(nll)

    # calculate errors
    param_errors = result.hesse()
    fit_result = uncertainties.correlated_values([x.value().numpy() for x in params], result.covariance())
    return fit_result


def zfit_linear(x, y, yerr):
    """
        Calculates a simple straight line fit given x,y and yerr.
        Returns slope, intercept as correlated ufloats.
        
        Intended for fitting to toyMC linearity tests.
    """
    # zfit treats the name as a global variable
    # if we want to loop several fits need to create a unique name for each parameter
    unique_id = str(uuid.uuid4())

    slope_param = zfit.Parameter('slope_' + unique_id, 1, -10, 10)
    intercept_param = zfit.Parameter('intercept_' + unique_id, 0, -1e4, 1e4)
    params = [slope_param, intercept_param]

    def chi2_loss(params):
        slope = params[0]
        intercept = params[1]
        y_pred = x * slope + intercept
        chi2 = ((y_pred - y) / yerr)**2
        return np.sum(chi2)

    loss = zfit.loss.SimpleLoss(chi2_loss, params, errordef=1)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(loss)
    param_errors = result.hesse()

    fit_result = uncertainties.correlated_values([x.value().numpy() for x in params], result.covariance())
    return fit_result

