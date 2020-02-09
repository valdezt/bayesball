from scipy.special import gammaln
import numpy as np
from scipy.optimize import minimize

def loglike_betabinom(params, *args):
    '''
    This function is the log of a well-behaved beta-binomial pdf for all values
    of n and k.
    '''

    a, b = params[0], params[1]
    k = args[0] # the OVERALL conversions
    n = args[1] # the number of at-bats (AE)

    logpdf = gammaln(n+1) + gammaln(k+a) + gammaln(n-k+b) + gammaln(a+b) - \
     (gammaln(k+1) + gammaln(n-k+1) + gammaln(a) + gammaln(b) + gammaln(n+a+b))

    return -np.sum(logpdf)

def fit_beta_binom(init_params, *args):
    '''
    This will find the alpha and beta values of a beta-binomial distribution
    using maximum likelihood.

    https://stackoverflow.com/questions/54505173/finding-alpha-and-beta-of-beta-binomial-distribution-with-scipy-optimize-and-log
    '''

    return minimize(
        loglike_betabinom,
        x0=init_params,
        args=args,
        method='L-BFGS-B',
        options={'disp': True, 'maxiter':500}
    )
