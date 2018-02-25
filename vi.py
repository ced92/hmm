"""Variational Inference module.

This module implements example 10.1.3 on page 470 in Bishop's 'Pattern
Recognition and Machine Learning'. It is used to compare the exact posterior
with the factorized variational approximation.
"""

__author__ = 'Cedric Seger'

import numpy as np
from scipy.stats import gamma
from scipy.stats import norm


class GammaNormal(object):
    """Represents a GammaNormal posterior distribution.

    The GammaNormal is a conjugate prior for the parameter space
    of the Gaussian likelihood model. The posterior will therefore also be
    GammaNormal and this class is responsible for calculating this posterior
    distribution given a dataset and a set of priors.

    Parameters
    ----------
    dataX : np.array
        The dataset on which to fit the model
    alpha_0 : number
        Shape parameter of Gamma prior
    beta_0 : number
        Scale parameter of Gamma prior
    mu_0 : number
        Location parameter of Normal prior
    lambda_0 : number
        Precision parameter of Normal prior
    """
    def __init__(self, dataX, alpha_0, beta_0, mu_0, lambda_0):
        self.init_params(dataX, alpha_0, beta_0, mu_0, lambda_0)

    def init_params(self, dataX, alpha_0, beta_0, mu_0, lambda_0):
        """Solves for the analytical GammaNormal posterior."""
        n = dataX.shape[0]
        data_mean = np.mean(dataX)
        variance = np.var(dataX)
        # Post gamma dist. params
        temp = variance + lambda_0*np.square(data_mean - mu_0)/(lambda_0 + n)
        self.beta_n = beta_0 + (n/2) * temp
        self.alpha_n = alpha_0 + n/2 - 1/2
        # Post normal dist. params
        lambda_n = lambda_0 + n
        mu_n = (mu_0*lambda_0 + n*data_mean) / (lambda_0 + n)
        self.mu_n = mu_n
        self.lambda_n = lambda_n
        print("Gamma settings:\n a:{alpha}, b:{beta}".format(
                                                    alpha=self.alpha_n,
                                                    beta=self.beta_n))
        print("Norm settings:\n mu:{mu}, prec:{prec}".format(
                                                mu=self.mu_n,
                                                prec=self.lambda_n*1))

    def pdf(self, tau, mu):
        """Calculates the pdf of the GammaNormal distribution.

        Can be used for visualizing the density function of the posterior
        distribution.

        Parameters
        ----------
        tau : number
            A specific setting of parameter tau
        mu : number
            A specific setting of parameter mu
        """
        prob_tau = gamma.pdf(tau, a=self.alpha_n, scale=1/self.beta_n)
        prob_mu = norm.pdf(mu, self.mu_n, 1/(self.lambda_n*tau))
        return prob_mu * prob_tau


class GaussianVI(object):
    """Factorized variational approximation to Gaussian-Gamma posterior.

    Finds an approximation to the above GammaNormal class by assuming
    distribution over parameter space can be factorized.

    Parameters
    ----------
    dataX : np.array
        The dataset on which to fit the model
    alpha_0 : number
        Shape parameter of Gamma prior
    beta_0 : number
        Scale parameter of Gamma prior
    mu_0 : number
        Location parameter of Normal prior
    lambda_0 : number
        Precision parameter of Normal prior
    """
    def __init__(self, dataX, alpha_0, beta_0, mu_0, lambda_0):
        self.init_params(dataX, alpha_0, beta_0, mu_0, lambda_0)

    def init_params(self, dataX, alpha_0, beta_0, mu_0, lambda_0):
        """Initializes the parameters."""
        n = dataX.shape[0]
        data_mean = np.mean(dataX)
        self.n = n
        self.lambda_0 = lambda_0
        self.mu_0 = mu_0
        self.beta_0 = beta_0
        self.alpha_0 = alpha_0
        self.dataX = dataX
        self.exp_tau = np.random.normal(2, 1)
        self.mu_n = (lambda_0 * mu_0 + n * data_mean) / (lambda_0 + n)
        self.alpha_n = alpha_0 + n/2
        self.lambda_n = None
        self.beta_n = None

    def update_params(self):
        """Convenenience method for updating all parameters."""
        self.update_gaussian()
        self.update_gamma()

    def update_gaussian(self):
        """"Updates the parameters for the Gaussian mdodel."""
        lambda_n = (self.lambda_0 + self.n) * self.exp_tau
        self.lambda_n = lambda_n

    def update_gamma(self):
        """Updates the parameters for the Gamma distribution.

        This method can only be called after at least one call
        to update_gaussian method.
        """
        exp_u = self.mu_n
        exp_mu_sq = (exp_u ** 2) + 1/self.lambda_n
        left = self.lambda_0 * (exp_mu_sq - 2*self.mu_0*exp_u + self.mu_0**2)
        right = np.square(self.dataX) - 2*self.dataX*exp_u + exp_mu_sq
        right = np.sum(right)
        beta_n = self.beta_0 + 0.5 * (left + right)
        self.beta_n = beta_n
        self.exp_tau = self.alpha_n / beta_n

    def pdf(self, tau, mu):
        """Calculates the pdf of the variational distribution.

        Parameters
        ----------
        tau : float
            A specific setting of parameter tau
        mu : float
            A specific setting of parameter mu
        """
        if(self.lambda_n):
            prob_mu = norm.pdf(mu, self.mu_n, 1/(self.lambda_n))
        else:
            prob_mu = norm.pdf(mu, self.mu_0, 1/(self.lambda_0))
        if (self.beta_n):
            prob_tau = gamma.pdf(tau, a=self.alpha_n, scale=1/self.beta_n)
        else:
            prob_tau = gamma.pdf(tau, a=self.alpha_0, scale=1/self.beta_0)
        return prob_mu * prob_tau
