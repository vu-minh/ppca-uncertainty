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
from matplotlib import pyplot as plt
from common.dataset import dataset
from common.plot.scatter import imscatter
import math
import argparse


dataset.set_data_home("./data")
writer = SummaryWriter()


class PPCADecoder(nn.Module):
    def __init__(self, data_dim=784, z_dim=2, hidden_dim=300):
        super(PPCADecoder, self).__init__()

        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, data_dim)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, z):
        hidden = self.softplus(self.fc1(z))
        loc_img = self.softplus(self.fc21(hidden))
        return loc_img


def ppca_model(data, hidden_dim=200, z_dim=2):
    N, D = data.shape
    H, M = hidden_dim, z_dim

    decoder_module = PPCADecoder(data_dim=D, z_dim=M, hidden_dim=H)
    pyro.module("ppca_decoder", decoder_module, update_module_params=True)

    sigma = pyro.sample(
        "sigma", LogNormal(torch.zeros(1, D), torch.ones(1, D)).to_event(1)
    )

    with pyro.plate("data_plate", N):
        Z = pyro.sample("Z", Normal(torch.zeros(N, M), torch.ones(N, M)).to_event(1))

        generated_X_mean = decoder_module.forward(Z)
        generated_X_scale = torch.ones(N, D) * sigma

        obs = pyro.sample(
            "obs", Normal(generated_X_mean, generated_X_scale).to_event(1), obs=data
        )


def trainVI(
    data, hidden_dim, learning_rate=1e-03, n_iters=500, trace_embeddings_interval=20
):
    pyro.enable_validation(True)
    pyro.set_rng_seed(0)
    pyro.clear_param_store()

    model = partial(ppca_model, hidden_dim=hidden_dim, z_dim=2)
    guide = AutoGuideList(model)
    guide.add(AutoDiagonalNormal(model=poutine.block(model, expose=["sigma"])))
    guide.add(AutoDiagonalNormal(model=poutine.block(model, expose=["Z"]), prefix="qZ"))

    optim = Adam({"lr": learning_rate})
    svi = SVI(model, guide, optim, loss=Trace_ELBO())

    fig_title = f"lr={learning_rate}/hidden-dim={hidden_dim}"
    data = torch.tensor(data, dtype=torch.float)
    for n_iter in tqdm(range(n_iters)):
        loss = svi.step(data)
        if n_iter % 10 == 0:
            writer.add_scalar("train_vi/loss", loss, n_iter)

        if trace_embeddings_interval and n_iter % trace_embeddings_interval == 0:
            z2d_loc = pyro.param("qZ_loc").reshape(-1, 2).data.numpy()
            fig = get_fig_plot_z2d(z2d_loc, fig_title)
            writer.add_figure("train_vi/z2d", fig, n_iter)

    # show named rvs
    print("List params and their size: ")
    for p_name, p_val in pyro.get_param_store().items():
        print(p_name, p_val.shape)

    z2d_loc = pyro.param("qZ_loc").reshape(-1, 2).data.numpy()
    z2d_scale = pyro.param("qZ_scale").reshape(-1, 2).data.numpy()
    fig = get_fig_plot_z2d(z2d_loc, fig_title)
    writer.add_figure("train_vi/z2d", fig, n_iters)
    return z2d_loc, z2d_scale


def get_fig_plot_z2d(z2d, title):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_title(title)
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
        tag=args.dataset_name,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DeepPPCAModel")
    ap.add_argument("-d", "--dataset_name", default="")
    ap.add_argument("-x", "--dev", action="store_true")
    ap.add_argument("-lr", "--learning_rate", default=1e-3, type=float)
    ap.add_argument("-s", "--scale_data", default="unitScale")
    ap.add_argument("-n", "--n_iters", default=1000, type=int)
    ap.add_argument("-hd", "--hidden_dim", default=300, type=int)

    args = ap.parse_args()
    writer.add_text(f"Params", str(args))

    X_original, X, y = dataset.load_dataset(
        args.dataset_name, preprocessing_method=args.scale_data
    )
    z2d_loc, z2d_scale = trainVI(
        data=X,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        n_iters=5 if args.dev else args.n_iters,
        trace_embeddings_interval=100,
    )

    # create_tensorboard_embedding(data=X_original, labels=y, features=z2d_loc)
