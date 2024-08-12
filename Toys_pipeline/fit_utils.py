# +
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a supplimental script used by the toys pipeline
"""

import numpy as np
import uncertainties
from uncertainties import ufloat

from iminuit import cost, Minuit
from scipy.stats import norm

def gauss(x, mu, sigma):
    return norm.pdf(x, mu, sigma)

def polyno(x, par):
    return np.polyval(par, x)  # for len(par) == 2, this is a line

def line(x, x0, x1):
    return x0 + x1*x

def minuit_gauss(x):
    # get starting values:
    mean = np.mean(x)
    std = np.std(x)

    # cost function and minuit
    cost_gauss = cost.UnbinnedNLL(data=x, pdf=gauss)
    m_gauss = Minuit(fcn=cost_gauss, mu=round(mean,1), sigma=round(std,1))
    m_gauss.migrad()
    
    # fit result
    result = uncertainties.correlated_values(m_gauss.values, m_gauss.covariance)
    # correlated_values will keep the correlation between mu, sigma
    return result # mu, sigma


def minuit_linear(x, y, yerr):
    # get starting values
    p = np.polynomial.Polynomial.fit(x, y, deg=1)
    y_intercept, slope = p.convert().coef
    
    # cost function and minuit
    cost_poly = cost.LeastSquares(x,y,yerr,model=polyno,loss='soft_l1')
    m_line = Minuit(cost_poly, (round(y_intercept,1),round(slope,1)) )
    m_line.migrad()
    
    # fit result
    result = uncertainties.correlated_values(m_line.values, m_line.covariance)
    return result # slope, y_int if model==polyno; y_int, slope if model==line


# import zfit
# import zfit.z.numpy as znp
# import uuid

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

