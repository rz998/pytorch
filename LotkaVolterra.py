import numpy as np
from scipy.stats import lognorm
from scipy.integrate import odeint

class LV():

       def __init__(self, T):
        # number of unknown (prior) parameters
        self.d = 4
        # prior parameters
        self.alpha_mu  = -0.125
        self.alpha_std = 0.5
        self.beta_mu   = -3
        self.beta_std  = 0.5
        self.gamma_mu  = -0.125
        self.gamma_std = 0.5
        self.delta_mu  = -3
        self.delta_std = 0.5
        # initial condition
        self.x0 = [30,1]
        # length of integration window
        self.T = T
        # observation parameters
        self.obs_std = np.sqrt(0.1)

       def sample_prior(self, N):
        # generate Normal samples
        alpha = lognorm.rvs(scale=np.exp(self.alpha_mu), s=self.alpha_std, size=(N,))
        beta  = lognorm.rvs(scale=np.exp(self.beta_mu),  s=self.beta_std,  size=(N,))
        gamma = lognorm.rvs(scale=np.exp(self.gamma_mu), s=self.gamma_std, size=(N,))
        delta = lognorm.rvs(scale=np.exp(self.delta_mu), s=self.delta_std, size=(N,))
        # join samples
        return np.vstack((alpha, beta, gamma, delta)).T

       def ode_rhs(self, z, t, theta):
        # extract parameters
        alpha, beta, gamma, delta = theta
        # compute RHS of
        fz1 = alpha * z[0] - beta * z[0]*z[1]
        fz2 = -gamma * z[1] + delta * z[0]*z[1]
        return np.array([fz1, fz2])

       def simulate_ode(self, theta, tt):
        # check dimension of theta
        assert(theta.size == self.d)
        # numerically intergate ODE
        return odeint(self.ode_rhs, self.x0, tt, args=(theta,))

       def sample_data(self, theta):
        # check inputs
        if len(theta.shape) == 1:
            theta = theta[np.newaxis,:]
        assert(theta.shape[1] == self.d)
        # define observation locations
        tt = np.arange(0, self.T, step=2)
        nt = 2*(len(tt)-1)
        # define arrays to store results
        xt = np.zeros((theta.shape[0], nt))
        # run ODE for each parameter value
        for j in range(theta.shape[0]):
            yobs = self.simulate_ode(theta[j,:], tt);
            # extract observations, flatten, and add noise
            yobs = np.abs(yobs[1:,:]).ravel()
            #xt[j,:] = lognorm.rvs(scale=np.exp(np.log(yobs)), s=self.obs_std, size=(1,))
            xt[j,:] = np.array([lognorm.rvs(scale=x, s=self.obs_std) for x in yobs])
        return (xt, tt)

       def log_prior_pdf(self, theta):
        # check dimensions of inputs
        assert(theta.shape[1] == self.d)
        # compute mean and variance
        prior_mean = [self.alpha_mu, self.beta_mu, self.gamma_mu, self.delta_mu]
        prior_std = [self.alpha_std, self.beta_std, self.gamma_std, self.delta_std]
        # evaluate product of PDFs for independent variables
        return np.sum(lognorm.logpdf(theta, scale=np.exp(prior_mean), s=prior_std), axis=1)

       def prior_pdf(self, theta):
        return np.exp( self.log_prior_pdf(theta) )



