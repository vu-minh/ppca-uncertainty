""" Reproduce figure 12.9 Bishop's PRML
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

from matplotlib import pyplot as plt
import seaborn as sns

from common.plot.scatter import config_font_size

PLOT_DIR = "./plots/esann2019"

# note: working with all distribution in 1D and 2D


def plot_dist(samples, label="p(?)", name=None):
    sns.distplot(samples, label=label)
    plt.legend()
    plt.savefig(f"{PLOT_DIR}/{name or label}.png", bbox_inches="tight")


def plot_ppca_model_2D(N=1000, D=2, M=1):
    # z: [N, M], M = 1
    p_z = Normal(loc=0.0, scale=1.0)
    z = p_z.sample(sample_shape=(N,))

    # w: [M, D], M = 1, D = 2
    p_w = Normal(loc=torch.zeros([M, D]), scale=torch.ones([M, D])).to_event(D)
    w = p_w.sample()
    print(w)

    plot_dist(z, label="p(z)", name="z_hist1D")


if __name__ == "__main__":
    pyro.set_rng_seed(2019)
    n_samples = 1000
    config_font_size(min_size=12)

    plot_ppca_model_2D()
