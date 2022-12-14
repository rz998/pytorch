import torch
import torch.nn as nn
import numpy as np
import scipy.io
import h5py
from sklearn.neighbors import KernelDensity

# fully connected feedforward neural network
class fcfnn(nn.Module):
    def __init__(self, layers, activ, activ_params=None, out_activ=None, out_activ_params=None):
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
     
    
# 2-D KDE
def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs):
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins,
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)


# Unit Gaussian Normalizer
class UnitGaussianNormalizer(nn.Module):
    def __init__(self, x=None, size=1, eps=1e-8):
        super(UnitGaussianNormalizer, self).__init__()

        assert (x is not None) or (size is not None), "Input or size must be specified."

        if x is not None:
            mean = torch.mean(x, 0).view(-1)
            std = torch.std(x, 0).view(-1)
        else:
            mean =  torch.zeros(size)
            std = torch.ones(size)

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        
        self.eps = eps

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.mean) / (self.std + self.eps)
        x = x.view(s)

        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x * (self.std + self.eps)) + self.mean
        x = x.view(s)

        return x

    def forward(self, x):
        return self.encode(x)


# MATLAB data reader
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float