"""25/02/2019: Can we visualize the uncertainty in the result of PPCA model?
"""

import torch
import numpy as np
from matplotlib import pyplot as plt

import mlflow
# from sklearn.decomposition import PCA

from common.dataset import dataset
import common.plot.simple_plot
from common.plot.simple_plot import SimpleFigure
from common.plot.scatter import imscatter, ellipse_scatter
from common.metric.dr_metrics import DRMetric
from ppca_model import MAP, generate


def scatter_with_images(fig, z2d, z2d_scale, original_data, labels_true, name):
    def _scatter():
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.scatter(z2d[:, 0], z2d[:, 1])
        imscatter(ax, z2d, data=original_data, zoom=.75, inverse_cmap=True,
                  custom_cmap=None, labels_true=labels_true)
        # ellipse_scatter(ax, z2d, z2d_scale, labels_true)

    fig.size(None).name(name).plot(_scatter)
    plt.close()


def scatter_with_errors(fig, z2d_loc, z2d_scale, labels_true, name, K=10):
    def _scatter():
        _, ax = plt.subplots(1, 1, figsize=(10, 10))

        # for each point in 2D, use the norm of variance (scale_x and scale_y)
        # as the 'error' and show them in the viz
        errors = 2000 * np.linalg.norm(z2d_scale, axis=1)

        # draw variance of each points by an ellipse
        ellipse_scatter(ax, z2d_loc, 5 * z2d_scale, labels_true, alpha=0.1)

        # get top 10 errors and highlight them
        ind = np.argpartition(errors, -K)[-K:]
        highlights = z2d_loc[ind]
        ax.scatter(highlights[:, 0], highlights[:, 1], c='w',
                   linewidths=2., edgecolors='red', s=errors[ind])

        # scatter the points with size of each point is its norm of variance
        ax.scatter(z2d_loc[:, 0], z2d_loc[:, 1], c=labels_true, s=errors,
                   alpha=0.5, linewidths=0, edgecolors='w', cmap='tab10')

    fig.size(None).name(name).plot(_scatter)
    plt.close()


def run_ppca(X_original, X, y, learning_rate, n_iters, plot_args={}):
    # test run ippca model with moved points
    moved_points = {
        5: [0.0, -2.0]
    }
    sigma_fix = 1e-4
    losses, z2d_loc, z2d_scale, W, sigma = MAP(
        X, learning_rate, n_iters,
        moved_points=moved_points, sigma_fix=sigma_fix)

    sum_vars = np.linalg.norm(z2d_scale, axis=1).sum()
    mlflow.log_metric('sum_vars', sum_vars)

    auc_rnx = DRMetric(X, z2d_loc).auc_rnx()
    mlflow.log_metric('aux_rnx', auc_rnx)

    fig = SimpleFigure(**plot_args)
    fig.size((6, 3)).name('losses').plot(lambda: plt.plot(losses))

    scatter_with_images(fig, z2d_loc, z2d_scale, X_original, y, name='z2d')
    scatter_with_errors(fig, z2d_loc, z2d_scale, y, 'z2d_error')

    # test generate
    z = torch.tensor([[0.0, 0.0]])
    x = generate(z, W, sigma)
    print(x.shape)


def nested_run(n_iters, learning_rates, datasets, preprocessing_method):
    for dataset_name in datasets:
        with mlflow.start_run():
            mlflow.log_param('dataset_name',
                             f"{dataset_name}_{preprocessing_method}")
            mlflow.log_param('n_iters', n_iters)

            X_original, X, y = dataset.load_dataset(dataset_name,
                                                    preprocessing_method)
            for lr in learning_rates:
                plot_args = {
                    'prefix': f'{dataset_name}-{lr}-{n_iters}',
                    'suffix': '.png',
                    'save_to_file': SAVE_FIGURES,
                    'track_with_mlflow': TRACK_FLOW
                }
                with mlflow.start_run(nested=True):
                    mlflow.log_param('lr', lr)
                    run_ppca(X_original, X, y, learning_rate=lr,
                             n_iters=n_iters, plot_args=plot_args)


def simple_run(dataset_name, n_iters, lr):
    mlflow.log_param('dataset_name', dataset_name)
    mlflow.log_param('n_iters', n_iters)
    mlflow.log_param('lr', lr)
    plot_args = {
        'prefix': f'{dataset_name}-{lr}-{n_iters}',
        'suffix': '.png',
        'save_to_file': SAVE_FIGURES,
        'track_with_mlflow': TRACK_FLOW,
    }

    X_original, X, y = dataset.load_dataset(dataset_name)
    run_ppca(X_original, X, y, lr, n_iters, plot_args)


if __name__ == '__main__':
    common.plot.simple_plot.PLOT_DIR = './plots'
    dataset.set_data_home('./data')

    SAVE_FIGURES = True
    TRACK_FLOW = True

    # mlflow.set_experiment('scikit-learn')
    # mlflow.log_param('dataset_name', dataset_name)
    # run_with_sklearn(data, original_data=X, labels_true=y)

    mlflow.set_experiment('PPCA_pyro')

#    learning_rates = [0.005, 0.01, 0.02, 0.025, 0.005, 0.075, 0.1, 0.15, 0.2]
#    n_iters = 350
#    datasets = ['FASHION200']
#    preprocessing_method = 'standardize' # 'no_preprocess'
#    nested_run(n_iters, learning_rates, datasets, preprocessing_method)

    simple_run(dataset_name='FASHION200', n_iters=250, lr=0.2)
