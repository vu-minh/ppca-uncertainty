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


def ppca_model(data, M=2, moved_points={}, sigma_fix=1e-3):
    N, D = data.shape

    # sigma: scale of data, (1, D)
    sigma = pyro.sample(
        "sigma", LogNormal(torch.zeros(1, D), torch.ones(1, D)).to_event(1)
    )

    # alpha: D-dim vector scale of one row of W (note, W: (M, D))
    alpha = pyro.sample(
        "alpha", LogNormal(torch.zeros(M, 1), torch.ones(M, 1)).to_event(1)
    )

    # W: (M, D): mapping from low-dim. to high-dim.
    W = pyro.sample(
        "W", Normal(torch.zeros(M, D), torch.ones(M, D) * alpha).to_event(2)
    )

    # Z: (N, M), X: (N, D)
    z_loc = torch.zeros(N, M)
    z_scale = torch.ones(N, M)

    if len(moved_points) > 0:
        for moved_id, (px, py) in moved_points.items():
            z_loc[moved_id, 0] = px
            z_loc[moved_id, 1] = py
            z_scale[moved_id] = sigma_fix

    with pyro.plate("data_loop", N):
        Z = pyro.sample("Z", Normal(z_loc, z_scale).to_event(1))

        pyro.sample(
            "obs", Normal(Z @ W, torch.ones(N, D) * sigma).to_event(1), obs=data
        )


def define_guide(model, param_guide=AutoDelta, latent_guide=AutoDiagonalNormal):
    guide = AutoGuideList(model)
    guide.add(
        param_guide(poutine.block(model), expose=["sigma", "alpha", "W"], prefix="q")
    )
    guide.add(latent_guide(poutine.block(model), expose=["Z"], prefix="qZ"))
    return guide


def train(data, model, guide, learning_rate=1e-3, n_iters=250):
    pyro.set_rng_seed(0)
    pyro.clear_param_store()

    optim = Adam({"lr": learning_rate})
    svi = SVI(model, guide, optim, loss=Trace_ELBO())

    losses = []
    for j in tqdm(range(n_iters)):
        loss = svi.step(data)
        losses.append(loss)

    # for name, value in pyro.get_param_store().items():
    #    print(name, value.shape)

    z2d_loc = pyro.param("qZ_loc").reshape(-1, 2).data.numpy()
    z2d_scale = pyro.param("qZ_scale").reshape(-1, 2).data.numpy()
    W = []  # pyro.param('q_W').data.numpy()
    sigma = []  # pyro.param('q_sigma').data.numpy()
    return losses, z2d_loc, z2d_scale, W, sigma


def MAP(data, learning_rate=1e-3, n_iters=250, moved_points={}, sigma_fix=1e-3):
    from functools import partial

    ippca_model = partial(
        ppca_model, M=2, moved_points=moved_points, sigma_fix=sigma_fix
    )
    data = torch.tensor(data, dtype=torch.float)
    auto_guide = define_guide(
        model=ippca_model, param_guide=AutoDelta, latent_guide=AutoDiagonalNormal
    )
    return train(
        data,
        model=ippca_model,
        guide=auto_guide,
        learning_rate=learning_rate,
        n_iters=n_iters,
    )


def generate(z, W, sigma, n_samples=100):
    # try to generate x from z
    # repeat z `n_samples` times
    M, D = W.shape
    samples = pyro.sample(
        "p_x",
        Normal(
            loc=z.repeat(n_samples, 1) @ torch.tensor(W),
            scale=torch.ones(n_samples, D) * torch.tensor(sigma),
        ),
    )
    return samples.mean(dim=0).data.numpy()
