"""Build PPCA model with pyro
"""
import time
import math
import argparse

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

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from common.dataset import dataset
from common.plot.scatter import imscatter
from common.metric.dr_metrics import DRMetric

from icommon import generate_noise


class PPCADecoder(nn.Module):
    def __init__(self, data_dim=784, z_dim=2, hidden_dim=300):
        super(PPCADecoder, self).__init__()

        self.hidden_dim = hidden_dim
        if hidden_dim > 0:
            self.fc1 = nn.Linear(z_dim, hidden_dim)
            self.fc21 = nn.Linear(hidden_dim, data_dim)
            self.softplus = nn.Softplus()
            self.sigmoid = nn.Sigmoid()
            self.tanh = nn.Tanh()
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.fc0 = nn.Linear(z_dim, data_dim)

    def forward(self, z):
        if self.hidden_dim > 0:
            hidden = self.softplus(self.fc1(z))
            hidden_dropout = self.dropout(hidden)
            loc_img = self.sigmoid(self.fc21(hidden_dropout))
        else:
            loc_img = self.fc0(z)
        return loc_img


def ppca_model(data, hidden_dim=200, z_dim=2, moved_points={}, sigma_fix=1e-3):
    N, D = data.shape
    H, M = hidden_dim, z_dim

    decoder_module = PPCADecoder(data_dim=D, z_dim=M, hidden_dim=H)
    pyro.module("ppca_decoder", decoder_module, update_module_params=True)

    sigma = pyro.sample(
        "sigma", LogNormal(torch.zeros(1, D), torch.ones(1, D)).to_event(1)
    )

    z_loc = torch.zeros(N, M)
    z_scale = torch.ones(N, M)

    if len(moved_points) > 0:
        for moved_id, (px, py) in moved_points.items():
            z_loc[moved_id, 0] = px
            z_loc[moved_id, 1] = py
            z_scale[moved_id] = sigma_fix

    with pyro.plate("data_plate", N):
        Z = pyro.sample("Z", Normal(z_loc, z_scale).to_event(1))

        generated_X_mean = decoder_module.forward(Z)
        generated_X_scale = torch.ones(N, D) * sigma

        obs = pyro.sample(
            "obs", Normal(generated_X_mean, generated_X_scale).to_event(1), obs=data
        )


def trainVI(
    data,
    hidden_dim,
    learning_rate=1e-03,
    n_iters=500,
    trace_embeddings_interval=20,
    writer=None,
    moved_points={},
    sigma_fix=1e-3,
):
    """Train (Deep) PPCA model with VI.
    Using tensorboardX SummaryWriter to log to tensorboard.
    Logging the internal result by setting "trace_embeddings_interval"
    to the expected interval (50, 100, ...).
    Set it to value larger than "n_iters" to disable interval logging.
    """
    pyro.enable_validation(True)
    pyro.set_rng_seed(0)
    pyro.clear_param_store()

    model = partial(
        ppca_model,
        hidden_dim=hidden_dim,
        z_dim=2,
        moved_points=moved_points,
        sigma_fix=sigma_fix,
    )
    guide = AutoGuideList(model)
    guide.add(AutoDiagonalNormal(model=poutine.block(model, expose=["sigma"])))
    guide.add(AutoDiagonalNormal(model=poutine.block(model, expose=["Z"]), prefix="qZ"))

    optim = Adam({"lr": learning_rate})
    svi = SVI(model, guide, optim, loss=Trace_ELBO())

    fig_title = f"lr={learning_rate}/hidden-dim={hidden_dim}"
    metric = DRMetric(X=data, Y=None)
    data = torch.tensor(data, dtype=torch.float)

    for n_iter in tqdm(range(n_iters)):
        loss = svi.step(data)
        if writer and n_iter % 10 == 0:
            writer.add_scalar("train_vi/loss", loss, n_iter)

        if writer and n_iter % trace_embeddings_interval == 0:
            z2d_loc = pyro.param("qZ_loc").reshape(-1, 2).data.numpy()

            auc_rnx = metric.update(Y=z2d_loc).auc_rnx()
            writer.add_scalar("metrics/auc_rnx", auc_rnx, n_iter)

            fig = get_fig_plot_z2d(z2d_loc, fig_title + f", auc_rnx={auc_rnx:.3f}")
            writer.add_figure("train_vi/z2d", fig, n_iter)

    # show named rvs
    print("List params and their size: ")
    for p_name, p_val in pyro.get_param_store().items():
        print(p_name, p_val.shape)

    z2d_loc = pyro.param("qZ_loc").reshape(-1, 2).data.numpy()
    z2d_scale = pyro.param("qZ_scale").reshape(-1, 2).data.numpy()
    if writer:
        fig = get_fig_plot_z2d(z2d_loc, fig_title)
        writer.add_figure("train_vi/z2d", fig, n_iters)
    return z2d_loc, z2d_scale


def get_fig_plot_z2d(z2d, title, with_imgs=True):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    plt.title(title)

    axes[0].scatter(z2d[:, 0], z2d[:, 1], c=y, alpha=0.5, cmap="jet")
    if with_imgs:
        axes[1].scatter(z2d[:, 0], z2d[:, 1], s=1)
        imscatter(axes[1], z2d, X_original, zoom=0.3)
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


def run_with_sklearn(data, labels):
    for model in [PCA, TSNE]:
        Z = model(n_components=2).fit_transform(data)

        metric = DRMetric(data, Z)
        auc_rnx = metric.auc_rnx()

        fig = get_fig_plot_z2d(
            Z, title=f"sklearn/{model.__name__}, auc_rnx={auc_rnx:.3f}", with_imgs=True
        )
        fig.savefig(f"./plots/{args.dataset_name}_sklearn_{model.__name__}.png")
        plt.close(fig)


if __name__ == "__main__":
    dataset.set_data_home("./data")

    help_msg = """
        Run DeepPPCAModel with custom params:
        $ python ppca_model2.py -d "DIGITS" -hd 50 -lr 0.0075 -n 2000
    """
    ap = argparse.ArgumentParser(description=help_msg)
    ap.add_argument("-r", "--run_id", default="999")
    ap.add_argument("-d", "--dataset_name", default="")
    ap.add_argument("-x", "--dev", action="store_true")
    ap.add_argument("-lr", "--learning_rate", default=1e-3, type=float)
    ap.add_argument("-s", "--scale_data", default="unitScale")
    ap.add_argument("-n", "--n_iters", default=1000, type=int)
    ap.add_argument(
        "-hd",
        "--hidden_dim",
        default=0,
        type=int,
        help="Number of hidden units, defaults to 0 to simply do Z@W",
    )
    ap.add_argument("-ad", "--add_noise", action="store_true")

    args = ap.parse_args()

    time_str = time.strftime("%b%d/%H:%M:%S", time.localtime())
    log_dir = (
        f"runs{args.run_id}/{args.dataset_name}/{time_str}_"
        + f"lr{args.learning_rate}_h{args.hidden_dim}"
    )
    print(log_dir)

    writer = SummaryWriter(log_dir=log_dir)
    X_original, X, y = dataset.load_dataset(
        args.dataset_name, preprocessing_method=args.scale_data
    )
    if args.add_noise:
        X_original = generate_noise("s&p", X_original)
        X = X_original / 255.0

    # run_with_sklearn(X, y)

    writer.add_text(f"Params", str(args))
    z2d_loc, z2d_scale = trainVI(
        data=X,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        n_iters=5 if args.dev else args.n_iters,
        trace_embeddings_interval=200,
        writer=writer,
    )

    # create_tensorboard_embedding(data=X_original, labels=y, features=z2d_loc)
