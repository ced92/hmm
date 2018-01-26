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

class GaussianVI(object):
    """docstring for GaussianVI."""
    def __init__(self, dataX, alpha_0, beta_0, mu_0, lambda_0):
        self.init_params(dataX, alpha_0, beta_0, mu_0, lambda_0)
        # Perform one update step for initializing
        # self.update_gaussian()
        # self.update_gamma()

    def init_params(self, dataX, alpha_0, beta_0, mu_0, lambda_0):
        """Initializes VI"""
        n = dataX.shape[0]
        data_mean = np.mean(dataX)

        self.n = n
        self.lambda_0 = lambda_0
        self.mu_0 = mu_0
        self.beta_0 = beta_0
        self.alpha_0 = alpha_0
        self.dataX = dataX

        self.exp_tau = np.random.normal(2,1) # initial guess
        self.mu_n = (lambda_0 * mu_0 + n * data_mean) / (lambda_0 + n)
        self.alpha_n = alpha_0 + n/2

        self.lambda_n = None
        self.beta_n = None

    def update_params(self):
        """Convenenience method for updating parameters"""
        self.update_gaussian()
        self.update_gamma()

    def update_gaussian(self):
        """"Updates gaussian parameters"""
        lambda_n = (self.lambda_0 + self.n) * self.exp_tau
        self.lambda_n = lambda_n

    def update_gamma(self):
        """ Updates gamma parameters
        Must be called after at least one call to update_gaussian
        """
        exp_u = self.mu_n
        exp_mu_sq = (exp_u ** 2) + 1/self.lambda_n

        x_1 = self.lambda_0 * (exp_mu_sq - 2*self.mu_0*exp_u + self.mu_0**2)
        x_2 = np.square(self.dataX) - 2*self.dataX*exp_u + exp_mu_sq
        x_2 = np.sum(x_2)
        beta_n = self.beta_0 + 0.5 * (x_1 + x_2)

        self.beta_n = beta_n
        self.exp_tau = self.alpha_n / beta_n



    def pdf(self, tau, mu):
        """Calculates pdf value for certain mu and tau parameters"""
        if(self.lambda_n):
            prob_mu = norm.pdf(mu, self.mu_n, 1/(self.lambda_n))
        else:
            prob_mu = norm.pdf(mu, self.mu_0, 1/(self.lambda_0))
        if (self.beta_n): # Use updated parameters (beta_n only exists if having updated)
            prob_tau = gamma.pdf(tau,a=self.alpha_n,scale=1/self.beta_n)
        else: # Use initial parameters
            prob_tau = gamma.pdf(tau,a=self.alpha_0,scale=1/self.beta_0)

        return prob_mu * prob_tau
