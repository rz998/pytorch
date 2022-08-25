import torch
import torch.nn as nn
import math
import numpy as np
import argparse
from scipy.stats import gamma, norm

# torch seed
torch.manual_seed(0)


# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--monotone_param", type=float, default=0.01, help="monotone penalty constant")
parser.add_argument("--dataset", type=str, default='tanh_v1', help="one of: tanh_v1,tanh_v2,tanh_v3")
parser.add_argument("--n_train", type=int, default=50000, help="number of training samples")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs")
parser.add_argument("--n_layers", type=int, default=3, help="number of layers in network")
parser.add_argument("--n_units", type=int, default=128, help="number of hidden units in each layer")
parser.add_argument("--batch_size", type=int, default=100, help="batch size (Should divide Ntest)")
parser.add_argument("--learning_rate", type=float, default=0.0002, help="learning rate")
args = parser.parse_args()


# tanh example
class tanh_v1():

    """
    Prior: y ~ U[-3,3]
    u = tanh(y) + Gamma[1,0.3]
    """

    def __init__(self):
        self.y_a= -3.
        self.y_b= 3.
        self.u_alpha = 1.
        self.u_beta  = 1./0.3

    def sample_prior(self, N):
        return np.random.uniform(self.y_a, self.y_b, (N,1))

    def sample_data(self, y):
        N = y.shape[0]
        g = gamma.rvs(self.u_alpha, loc=0, scale=1./self.u_beta, size=(N,1))
        return np.tanh(y) + g

    def likelihood_function(self, y, u):
        g = u - np.tanh(y)
        return gamma.pdf(g, self.u_alpha, loc=0, scale=1./self.u_beta)


# generate data
if  args.dataset == 'tanh_v1':
    pi = tanh_v1()
else:
    print('dataset not supported')

y_train=pi.sample_prior(args.n_train)
u_train=pi.sample_data(y_train)

y_train = torch.from_numpy(y_train.astype(np.float32))
u_train = torch.from_numpy(u_train.astype(np.float32))

dim_y = y_train.shape[1]
dim_u = u_train.shape[1]


# batch
bsize = args.batch_size
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(y_train, u_train), batch_size=bsize, shuffle=True)
ydata_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(y_train, ), batch_size=bsize, shuffle=True)


# loss function
mse_loss = torch.nn.MSELoss()


# fully connected feedforward neural network
class fcfnn(nn.Module):
    def __init__(self, layers, activ, activ_params, out_activ, out_activ_params):
        super(fcfnn, self).__init__()
        self.n_layers=len(layers)-2

        self.layers=nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(layers[i],layers[i+1]))
            if activ_params is not None:
                self.layers.append(activ(*activ_params))
            else:
                self.layers.append(activ())
        self.layers.append(nn.Linear(layers[self.n_layers],layers[self.n_layers+1]))

        if out_activ is not None:
            if out_activ_params is not None:
                self.layers.append(out_activ(*out_activ_params))
            else:
                self.layers.append(out_activ())


    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)
        return x

# build the networks
network_params = [dim_u+dim_y] + args.n_layers * [args.n_units]
## generator
G = fcfnn(network_params + [dim_u], nn.LeakyReLU, activ_params=[0.2, True])
## discriminator
D = fcfnn(network_params + [1], nn.LeakyReLU, activ_params=[0.2, True], out_activ_params=nn.Sigmoid)
