import numpy as np
from scipy.stats import gamma
from scipy.stats import norm

class GammaNormal(object):
    """Finds the GammaNormal posterior distribution"""
    def __init__(self, dataX, alpha_0, beta_0, mu_0, lambda_0):
        self.init_params(dataX, alpha_0, beta_0, mu_0, lambda_0)

    def init_params(self, dataX, alpha_0, beta_0, mu_0, lambda_0):
        n = dataX.shape[0]
        data_mean = np.mean(dataX)
        variance = np.var(dataX)

        # Post gamma dist. params
        temp = variance + lambda_0 * np.square(data_mean - mu_0) / (lambda_0 + n)
        self.beta_n = beta_0 + (n/2) * temp
        self.alpha_n = alpha_0 + n/2 - 1/2

        # Post normal dist. params
        lambda_n = lambda_0 + n
        mu_n = (mu_0*lambda_0 + n*data_mean) / (lambda_0 + n)
        self.mu_n = mu_n
        self.lambda_n = lambda_n

        print("Gamma settings: \n a:{}, b:{}".format(self.alpha_n, self.beta_n))
        print("Norm settings: \n mu:{}, prec:{}".format(self.mu_n, self.lambda_n*1))

    def pdf(self, tau, mu):
        prob_tau = gamma.pdf(tau,a=self.alpha_n,scale=1/self.beta_n)
        prob_mu = norm.pdf(mu, self.mu_n, 1/(self.lambda_n*tau))
        return prob_mu * prob_tau
