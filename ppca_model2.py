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
from functools import partial


class DeepPPCAModel(nn.Module):
    def __init__(self, n_data_dims, n_latent_dims=2):
        super(DeepPPCAModel, self).__init__()
        self.n_data_dims = n_data_dims
        self.n_latent_dims = n_latent_dims
        self.linear1 = nn.Linear(
            in_features=n_latent_dims, out_features=n_data_dims, bias=False
        )

    def forward(self, z):
        return self.linear1(z)


def deep_ppca_model(data, M=2, moved_points={}, sigma_fix=1e-3):
    N, D = data.shape

    deep_ppca = DeepPPCAModel(n_data_dims=D, n_latent_dims=M)

    alpha = pyro.sample(
        "alpha", LogNormal(torch.zeros(1, M), torch.ones(1, M)).to_event(2)
    )
    w_prior = pyro.distributions.Normal(
        torch.zeros(D, M), torch.ones(D, M) * alpha
    ).to_event(2)
    priors = {"linear1.weight": w_prior}

    lifted_module = pyro.random_module("deep_ppca", deep_ppca, priors)
    lifted_ppca_model = lifted_module()

    sigma = pyro.sample(
        "sigma", LogNormal(torch.zeros(1, D), torch.ones(1, D)).to_event(1)
    )

    with pyro.plate("data_plate", N):
        Z = pyro.sample("Z", Normal(torch.zeros(N, M), torch.ones(N, M)).to_event(1))

        generated_X_mean = lifted_ppca_model(Z)
        generated_X_scale = torch.ones(N, D) * sigma

        obs = pyro.sample(
            "obs", Normal(generated_X_mean, generated_X_scale).to_event(1), obs=data
        )
        return obs


def trainVI(data, learning_rate=1e-03, n_iters=500):
    pyro.enable_validation(True)
    pyro.set_rng_seed(0)
    pyro.clear_param_store()

    model = partial(deep_ppca_model, M=2)
    guide = AutoGuideList(model)
    guide.add(
        AutoDiagonalNormal(
            model=poutine.block(
                model, expose=["alpha", "sigma", "deep_ppca$$$linear1.weight"]
            ),
            prefix="q",
        )
    )
    guide.add(AutoDiagonalNormal(model=poutine.block(model, expose=["Z"])))

    optim = Adam({"lr": learning_rate})
    svi = SVI(model, guide, optim, loss=Trace_ELBO())

    losses = []
    data = torch.tensor(data, dtype=torch.float)
    for j in tqdm(range(n_iters)):
        loss = svi.step(data)
        losses.append(loss)

    # show named rvs
    # print("List params", pyro.get_param_store().keys())
    # List params dict_keys(['q_sigma', 'auto_loc', 'auto_scale'])

    z2d_loc = pyro.param("auto_loc").reshape(-1, 2).data.numpy()
    z2d_scale = pyro.param("auto_scale").reshape(-1, 2).data.numpy()
    return losses, z2d_loc, z2d_scale


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from common.dataset import dataset

    dataset.set_data_home("./data")

    # X_original, X, y = dataset.load_dataset("DIGITS", preprocessing_method="unitScale")
    _, X, y = dataset.load_dataset("DIGITS")
    losses, z2d_loc, z2d_scale = trainVI(data=X, learning_rate=0.25, n_iters=1000)

    plt.figure(figsize=(7, 3))
    plt.plot(losses)
    plt.yscale("log")
    plt.savefig("./temp_loss.png")

    plt.figure(figsize=(5, 5))
    plt.scatter(z2d_loc[:, 0], z2d_loc[:, 1], alpha=0.5, c=y, cmap="jet")
    plt.savefig("./temp_2d.png")
