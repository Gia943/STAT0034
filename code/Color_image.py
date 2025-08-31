import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time

import torch
!pip install pykeops
!pip install --upgrade geomloss
import pykeops
from geomloss import SamplesLoss
from pykeops.torch import generic_logsumexp


use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

from random import choices
import imageio
from matplotlib import pyplot as plt


def load_image(fname):
    img = imageio.imread(fname)  # RGB
    return img / 255.0  # Normalized to [0,1]


def RGB_cloud(fname, sampling, dtype=torch.FloatTensor):
    A = load_image(fname)
    A = A[::sampling, ::sampling, :]
    return torch.from_numpy(A).type(dtype).view(-1, 3)


def display_cloud(ax, x):
    x_ = x.detach().cpu().numpy()
    ax.scatter(x_[:, 0], x_[:, 1], x_[:, 2], s=25 * 500 / len(x_), c=x_)


def display_image(ax, x):
    W = int(np.sqrt(len(x)))
    x_ = x.view(W, W, 3).detach().cpu().numpy()
    ax.imshow(x_)

sampling = 8 if not use_cuda else 1

X_i = RGB_cloud("data/garden1.jpg", sampling, dtype)
Y_j = RGB_cloud("data/memorial2.jpg", sampling, dtype)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(2, 2, 1)
display_image(ax, X_i)
ax.set_title("Source image")
ax = fig.add_subplot(2, 2, 2)
display_image(ax, Y_j)
ax.set_title("Target image")

ax = fig.add_subplot(2, 2, 3, projection="3d")
display_cloud(ax, X_i)
ax.set_title("Source point cloud")
ax = fig.add_subplot(2, 2, 4, projection="3d")
display_cloud(ax, Y_j)
ax.set_title("Target point cloud")
plt.tight_layout()

def color_transfer(loss, lr=1):
    """Flows along the gradient of the loss function.

    Parameters:
        loss ((x_i,y_j) -> torch float number):
            Real-valued loss function.
        lr (float, default = 1):
            Learning rate, i.e. time step.
    """

    # Parameters for the gradient descent
    Nsteps = 11
    display_its = [1, 10]

    # Make sure that we won't modify the reference samples
    x_i, y_j = X_i.clone(), Y_j.clone()

    # We're going to perform gradient descent on Loss(α, β)
    # wrt. the positions x_i of the diracs masses that make up α:
    x_i.requires_grad = True

    t_0 = time.time()

    plt.figure(figsize=(12, 12))
    k = 3
    ax = plt.subplot(2, 2, 1)
    display_image(ax, X_i)
    ax.set_title("Source image")
    plt.xticks([], [])
    plt.yticks([], [])
    ax = plt.subplot(2, 2, 2)
    display_image(ax, Y_j)
    ax.set_title("Target image")
    plt.xticks([], [])
    plt.yticks([], [])

    for i in range(Nsteps):  # Euler scheme ===============
        # Compute cost and gradient
        L_αβ = loss(x_i, y_j)
        [g] = torch.autograd.grad(L_αβ, [x_i])

        if i in display_its:  # display
            ax = plt.subplot(2, 2, k)
            display_image(ax, x_i)
            ax.set_title("it = {}".format(i))
            k = k + 1
            plt.xticks([], [])
            plt.yticks([], [])

        # in-place modification of the tensor's values
        x_i.data -= lr * len(x_i) * g

    plt.title(
        "it = {}, elapsed time: {:.2f}s/it".format(i, (time.time() - t_0) / Nsteps)
    )
    plt.tight_layout()

from pykeops.torch import generic_logsumexp as _glse
import geomloss.sinkhorn_samples as _ss
_ss.generic_logsumexp = _glse 

from geomloss import SamplesLoss

color_transfer(SamplesLoss("sinkhorn", blur=0.3,backend="multiscale"))
color_transfer(SamplesLoss("sinkhorn", blur=0.1,backend="multiscale"))
color_transfer(SamplesLoss("sinkhorn", blur=0.9,backend="multiscale"))
color_transfer(SamplesLoss("sinkhorn", blur=0.1, reach=0.4,backend="multiscale"))

plt.show()

def color_transfer(loss, lr=1):
    Nsteps = 11   


    x_i, y_j = X_i.clone(), Y_j.clone()

    x_i.requires_grad = True

    for i in range(Nsteps):  
        L_αβ = loss(x_i, y_j)
        [g] = torch.autograd.grad(L_αβ, [x_i])

        x_i.data -= lr * len(x_i) * g

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1)
    display_image(ax, x_i)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()

color_transfer(SamplesLoss("sinkhorn", blur=0.9,backend="multiscale"))
