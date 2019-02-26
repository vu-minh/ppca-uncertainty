"""Build PPCA model with pyro
"""

import torch
from torch import nn
from torch.distributions import constraints

import pyro
from pyro import poutine
from pyro.distributions import Normal, LogNormal, Dirichlet, Categorical, Gamma
from pyro.contrib.autoguide import *
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate

from tqdm import tqdm


def ppca_model(data, M=2):
    N, D = data.shape

    # sigma: scale of data, (1, D)
    sigma = pyro.sample('sigma', LogNormal(
        torch.zeros(1, D), torch.ones(1, D)
    ).to_event(1))

    # alpha: D-dim vector scale of one row of W (note, W: (M, D))
    alpha = pyro.sample('alpha', LogNormal(
        torch.zeros(M, 1), torch.ones(M, 1)
    ).to_event(1))

    # W: (M, D): mapping from low-dim. to high-dim.
    W = pyro.sample('W', Normal(
        torch.zeros(M, D), torch.ones(M, D) * alpha
    ).to_event(2))


    # Z: (N, M), X: (N, D)
    with pyro.plate('data_loop', N):
        Z = pyro.sample('Z', Normal(
            torch.zeros(N, M), torch.ones(N, M)
        ).to_event(1))

        pyro.sample('obs', Normal(
            Z @ W, torch.ones(N, D) * sigma
        ).to_event(1), obs=data)


def define_guide(model, param_guide=AutoDelta, latent_guide=AutoDiagonalNormal):
    guide = AutoGuideList(model)
    guide.add(param_guide(poutine.block(
        model, expose=['sigma', 'alpha', 'W']), prefix='q'))
    guide.add(latent_guide(poutine.block(model, expose=['Z']), prefix='qZ'))
    return guide


def train(data, model, guide, learning_rate=1e-3, n_iters=250):
    pyro.set_rng_seed(0)
    pyro.clear_param_store()

    optim = Adam({'lr': learning_rate})
    svi = SVI(model, guide, optim, loss=Trace_ELBO())

    losses = []
    for j in tqdm(range(n_iters)):
        loss = svi.step(data)
        losses.append(loss)

    for name, value in pyro.get_param_store().items():
        print(name, value.shape)

    z2d_loc = pyro.param('qZ_loc').reshape(-1, 2).data.numpy()
    z2d_scale = pyro.param('qZ_scale').reshape(-1, 2).data.numpy()
    return losses, z2d_loc, z2d_scale


def MAP(data, learning_rate=1e-3, n_iters=250):
    data = torch.tensor(data, dtype=torch.float)
    auto_guide = define_guide(model=ppca_model, param_guide=AutoDelta,
                              latent_guide=AutoDiagonalNormal)
    return train(data, model=ppca_model, guide=auto_guide,
                 learning_rate=learning_rate, n_iters=n_iters)
