import numpy as np
from scipy.stats import gamma
from scipy.stats import norm

# def getGamma(dataX, mean_0, b_0, a_0, lambda_0, mean):
#     num_samples = dataX.shape[0]
#     param_a = a_0 + (num_samples + 1) / 2
#     dataX_sq = np.square(dataX)
#     param_b = 0.5 * (np.sum(dataX) - 2*mean*np.sum(dataX) + num_samples * mean ** 2)
#     param_b += lambda_0/2 * np.square(mean-mean_0) + b_0
#     print(param_a)
#     print(param_b)
#
#     rv = gamma(param_a, param_b)
#     return rv
#

class GammaNormal(object):
    """Finds the GammaNormal posterior distribution"""
    def __init__(self, dataX, alpha_0, beta_0, mu_0, lambda_0):
        self.init_params(dataX, alpha_0, beta_0, mu_0, lambda_0)

    def init_params(dataX, alpha_0, beta_0, mu_0, lambda_0):
        n = dataX.shape[0]
        data_mean = np.mean(dataX, axis=0)
        variance = np.var(dataX)

        # Post gamma dist. params
        temp = varaiance + lambda_0 * np.square(data_mean - mu_0) / (lambda_0 + n)
        self.beta_n = beta_0 + (n/2) * temp
        self.alpha_n = alpha_0 + n/2 - 1/2
        # self.gamma = gamma(alpha_n, beta_n)

        # Post normal dist. params
        lambda_n = lambda_0 + n
        mu_n = (mu_0*lambda_0 + n*data_mean) / (lambda_0 + n)
        self.mu_n = mu_n
        self.lambda_n = lambda_n
        # self.normal = norm(mu_n, ())

    def pdf(tau, mu):
        prob_tau = gamma.pdf(tau,self.alpha_n,scale=self.beta_n)
        prob_mu = norm.pdf(mu, self.mu_n, 1/(self.lambda_n*tau))

        return prob_mu * prob_tau
