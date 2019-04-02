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
from tensorboardX import SummaryWriter

writer = SummaryWriter()


class DeepPPCAModel(nn.Module):
    def __init__(self, n_data_dims, n_latent_dims=2):
        super(DeepPPCAModel, self).__init__()
        self.n_data_dims = n_data_dims
        self.n_latent_dims = n_latent_dims
        self.linear1 = nn.Linear(
            in_features=n_latent_dims, out_features=n_data_dims, bias=True
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
    bias_prior = pyro.distributions.Normal(
        torch.zeros(1, D), torch.ones(1, D)
    ).to_event(1)
    priors = {
        "linear1.weight": w_prior,
        # "linear1.bias": bias_prior
    }

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


def trainVI(data, learning_rate=1e-03, n_iters=500, trace_embeddings_interval=20):
    pyro.enable_validation(True)
    pyro.set_rng_seed(0)
    pyro.clear_param_store()

    model = partial(deep_ppca_model, M=2)
    guide = AutoGuideList(model)
    guide.add(
        AutoDiagonalNormal(
            model=poutine.block(
                model,
                expose=[
                    "alpha",
                    "sigma",
                    "deep_ppca$$$linear1.weight",
                    # "deep_ppca$$$linear1.bias",
                ],
            ),
            prefix="q",
        )
    )
    guide.add(AutoDiagonalNormal(model=poutine.block(model, expose=["Z"])))

    optim = Adam({"lr": learning_rate})
    svi = SVI(model, guide, optim, loss=Trace_ELBO())

    data = torch.tensor(data, dtype=torch.float)
    for n_iter in tqdm(range(n_iters)):
        loss = svi.step(data)
        writer.add_scalar("train_vi/loss", loss, n_iter)

        if trace_embeddings_interval and n_iter % trace_embeddings_interval == 0:
            z2d_loc = pyro.param("auto_loc").reshape(-1, 2).data.numpy()
            fig = get_fig_plot_z2d(z2d_loc)
            writer.add_figure("train_vi/z2d", fig, n_iter)

    # show named rvs
    # print("List params", pyro.get_param_store().keys())

    z2d_loc = pyro.param("auto_loc").reshape(-1, 2).data.numpy()
    z2d_scale = pyro.param("auto_scale").reshape(-1, 2).data.numpy()
    fig = get_fig_plot_z2d(z2d_loc)
    writer.add_figure("train_vi/z2d", fig, n_iters)
    return z2d_loc, z2d_scale


def get_fig_plot_z2d(z2d):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(z2d[:, 0], z2d[:, 1], alpha=0.5)
    imscatter(ax, z2d, X_original, zoom=0.5)
    return fig


def create_tensorboard_embedding(data, labels, features=None):
    N, D = data.shape
    img_size = int(math.sqrt(D))
    images = torch.tensor(data, dtype=torch.float)
    writer.add_embedding(
        features if features is not None else images,
        metadata=labels,
        label_img=images.view(N, 1, img_size, img_size),
        tag=dataset_name,
    )


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from common.dataset import dataset
    from common.plot.scatter import imscatter
    import math

    dataset.set_data_home("./data")
    dataset_name = "FASHION200"

    X_original, X, y = dataset.load_dataset(
        dataset_name, preprocessing_method="unitScale"
    )
    z2d_loc, z2d_scale = trainVI(
        data=X, learning_rate=0.1, n_iters=1000, trace_embeddings_interval=25
    )

    # create_tensorboard_embedding(data=X_original, labels=y, features=z2d_loc)
