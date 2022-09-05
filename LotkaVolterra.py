import numpy as np
from scipy.stats import lognorm
from scipy.integrate import odeint
import matplotlib.pyplot as plt

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
            xt[j,:] = np.array([lognorm.rvs(scale=x, s=self.obs_std) for x in yobs])
        return (xt, tt)


# # define model
# T = 20
# true_LV = LV(T)

# # define true parameters and observation
# xtrue = np.array([0.6859157, 0.10761319, 0.88789904, 0.116794825])
# tt = np.linspace(0,true_LV.T,1000)
# ytrue = true_LV.simulate_ode(xtrue, tt)
# yobs,tobs = true_LV.sample_data(xtrue)
# nobs = int(yobs.size/2.)
# yobs_plot = yobs.reshape((nobs,2))

# # plot single simulation
# plt.figure()
# plt.plot(tt,ytrue[:,0], 'blue', tt,ytrue[:,1],'red')
# plt.plot(tobs[1:],yobs_plot[:,0],'o', markersize=8, color='blue')
# plt.plot(tobs[1:],yobs_plot[:,1],'o', markersize=8, color='red')
# plt.xlabel('$t$')
# plt.ylabel('Observations')
# plt.show()



