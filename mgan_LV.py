import torch
import torch.nn as nn
import numpy as np
import math
from LotkaVolterra import LV
import argparse
import matplotlib.pyplot as plt
from utility import kde2D, fcfnn, UnitGaussianNormalizer

# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--monotone_param", type=float, default=0.01, help="monotone penalty constant")
parser.add_argument("--n_train", type=int, default=20000, help="number of training samples")
parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs")
parser.add_argument("--n_layers", type=int, default=3, help="number of layers in network")
parser.add_argument("--n_units", type=int, default=128, help="number of hidden units in each layer")
parser.add_argument("--batch_size", type=int, default=100, help="batch size (Should divide Ntest)")
parser.add_argument("--learning_rate", type=float, default=0.0002, help="learning rate")
args = parser.parse_args()

# set torch seed
torch.manual_seed(0)


# set device: cuda, cpu
device = torch.device('cpu')


# pick dataset
T = 20
pi = LV(T)


# generate data and normalize
u_train = pi.sample_prior(args.n_train)
y_train, _ = pi.sample_data(u_train)
y_train = torch.from_numpy(y_train.astype(np.float32))
u_train = torch.from_numpy(u_train.astype(np.float32))
u_normalizer = UnitGaussianNormalizer(u_train)
u_train = u_normalizer.encode(u_train)
y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

# get dimensions of data
dim_u = u_train.shape[1]
dim_y = y_train.shape[1]



# batch
bsize = args.batch_size
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(y_train, u_train), batch_size=bsize, shuffle=True)
ydata_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(y_train, ), batch_size=bsize, shuffle=True)

# loss function
mse_loss = torch.nn.MSELoss()

# build networks
G_params = [dim_u+dim_y] + [args.n_units] + [2 * args.n_units] + [4 * args.n_units]
D_params = [dim_u+dim_y] + [4 * args.n_units] + [2 * args.n_units] + [args.n_units]
# generator network
G = fcfnn(G_params + [dim_u], nn.LeakyReLU, activ_params=[0.2, True]).to(device)
# discriminator network
D = fcfnn(D_params + [1], nn.LeakyReLU, activ_params=[0.2, True], out_activ=nn.Sigmoid).to(device)


# optimizers
opt_G = torch.optim.Adam(G.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

# schedulers
sch_G = torch.optim.lr_scheduler.StepLR(opt_G, step_size = len(train_loader), gamma=0.995)
sch_D = torch.optim.lr_scheduler.StepLR(opt_D, step_size = len(train_loader), gamma=0.995)


# define arrays to store results
monotonicity    = torch.zeros(args.n_epochs,)
D_train         = torch.zeros(args.n_epochs,)
G_train         = torch.zeros(args.n_epochs,)

for ep in range(args.n_epochs):

    G.train()
    D.train()

    # define variable for batch losses
    D_batch = 0.0
    G_batch = 0.0
    mon_percent = 0.0

    for y, x in train_loader:

        # data batch
        y, x = y.to(device), x.to(device)

        ones = torch.ones(bsize, 1, device=device)
        zeros = torch.zeros(bsize, 1, device=device)

        # reset gradient of G

        opt_G.zero_grad()

        # draw reference sample
        z1 = next(iter(ydata_loader))[0].to(device)
        z2 = torch.randn(bsize, dim_u, device=device)
        z = torch.cat((z1, z2), 1)

        # transport reference to conditional (z1 is transported by identity map)
        Gz = G(z)


        # compute loss for generator
        G_loss = mse_loss(D(torch.cat((z1, Gz), 1)), ones)
        G_batch += G_loss.item()

        # draw new reference sample
        z1_prime = next(iter(ydata_loader))[0].to(device)
        z2_prime = torch.randn(bsize, dim_u, device=device)
        z_prime = torch.cat((z1_prime, z2_prime), 1)


        # monotonicity penalty
        mon_penalty = torch.sum(((Gz - G(z_prime)).view(bsize,-1))*((z2 - z2_prime).view(bsize,-1)), 1)
        if args.monotone_param > 0.0:
            G_loss = G_loss - args.monotone_param*torch.mean(mon_penalty)

        # take step for F
        G_loss.backward()
        opt_G.step()
        sch_G.step()

        # percent of examples in batch with monotonicity satisfied
        mon_penalty = mon_penalty.detach() + torch.sum((z1.view(bsize,-1) - z1_prime.view(bsize,-1))**2, 1).detach()
        mon_percent += float((mon_penalty>=0).sum().item())/bsize

        # loss for discriminator

        opt_D.zero_grad()

        # compute loss for discriminator
        D_loss = 0.5*(mse_loss(D(torch.cat((y,x),1)), ones) + mse_loss(D(torch.cat((z1, Gz.detach()), 1)), zeros))
        D_batch += D_loss.item()

        # take step for D
        D_loss.backward()
        opt_D.step()
        sch_D.step()


    G.eval()
    D.eval()

    # average monotonicity percent over batches
    mon_percent = mon_percent/math.ceil(float(args.n_train)/bsize)
    monotonicity[ep] = mon_percent

    # average generator and discriminator losses over batches
    G_train[ep] = G_batch/math.ceil(float(args.n_train)/bsize)
    D_train[ep] = D_batch/math.ceil(float(args.n_train)/bsize)

    print('Epoch %3d, Monotonicity: %f, G loss: %f, D loss: %f' % \
         (ep, monotonicity[ep], G_train[ep], D_train[ep]))


# plot

# define true parameters and observation
true_LV = LV(T)
utrue = np.array([0.83194674, 0.04134147, 1.0823151, 0.03991483])
tt = np.linspace(0,true_LV.T,1000)
ytrue = true_LV.simulate_ode(utrue, tt)
yobs,tobs = true_LV.sample_data(utrue)

# generate test samples
Ntest = 1000
yobs = torch.from_numpy(yobs.astype(np.float32))
yobs = y_normalizer.encode(yobs)
yi = yobs.repeat(Ntest,1).to(device)
z = torch.randn(Ntest, dim_u, device=device)
with torch.no_grad():
    Gz = G(torch.cat((yi, z), 1))
Gz = u_normalizer.decode(Gz)
Gz = Gz.cpu().numpy()

# plot joint density with KDE

plt.figure()
xx, yy, zz = kde2D(Gz[:,0], Gz[:,1], 0.05)
plt.contourf(xx, yy, np.exp(zz), cmap='Blues')
plt.plot(utrue[0], utrue[1], 'o', markersize=5, color='red')
plt.show()