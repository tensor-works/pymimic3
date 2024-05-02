import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import pdb
import seaborn as sn
import tensorflow as tf
from pathlib import Path
from utils import make_prediction_vector


def plot_fourier_transform(series, years):
    """_summary_

    Args:
        series (_type_): _description_
    """
    fast_fourier_transform = tf.signal.rfft(series)
    f_per_dataset = np.arange(0, len(fast_fourier_transform))

    f_per_year = f_per_dataset / years
    print(max(np.abs(fast_fourier_transform)))
    # pdb.set_trace()
    plt.step(f_per_year, np.abs(fast_fourier_transform))
    plt.xscale('log')
    plt.ylim(0, max(np.abs(fast_fourier_transform)) / 2)
    plt.xlim([0.1, max(plt.xlim())])
    plt.xticks([1, 12, 52.2, 365.2524, 365.2524 * 2, 365.2524 * 4, 365.2524 * 24],
               labels=['1/Year', '1/Month', '1/Week', '1/day', '1/12', '1/6', '1/hour'])
    _ = plt.xlabel('Frequency (log scale)')
    plt.title("Fast Fourier Transform")


def _subplot_generator(data_df, name, features, layout, type, save=False):
    """
    This function generates subplots of the specified type with the specified parameters.

    Parameters:
        data_df:    data frame containing the data which is to be plotted.
        name:       name under which to save the plot
        features:   columns from the frame which are to be plotted
        layout:     of the plot in rows x columns
        type:       plot type
    
    Returns:
        fig:    matplotlib generated figure obj
        axs:    obj 
    """

    if layout[0] * layout[1] < len(features):
        print(
            f"Layout not valid, there are {len(features)} columns within the data frame and only {layout[0] * layout[1]} subplots!"
        )
        return

    if not name:
        name = "Subplots"
    if layout[1] > len(features):
        ncols = len(features)
    else:
        ncols = layout[1]
    nrows = int(np.ceil(len(features) / layout[1]))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 7, nrows * 3.6))
    for index, feature in enumerate(features):
        rows = int(np.floor(index / layout[1]))
        cols = index % layout[1]
        if nrows == ncols == 1:
            axs.plot(data_df[feature])
            axs.set_title(feature)
        elif nrows == 1:
            getattr(axs[cols], type)(data_df[feature])
            axs[cols].set_title(feature)
        else:
            getattr(axs[rows, cols], type)(data_df[feature])
            axs[rows, cols].set_title(feature)

    if save:
        fig.savefig(Path(plot_location, f"{name}.png"), dpi=200)

    return fig, axs


def plot(data_df, name="", features=None, subplots=True, layout=(27, 2), save=False):
    """
    Plots against index.

    Parameters:
        data_df:    data frame containing the data which is to be plotted.
        name:       name under which to save the plot
        features:   columns from the frame which are to be plotted
        layout:     of the plot in rows x columns
    
    Returns:
        fig:    matplotlib generated figure obj
        axs:    obj 
    """
    plt.clf()
    # implement something to adjust layout ratio if fewer features where passed
    if isinstance(data_df, pd.core.series.Series):
        features = [data_df.name]
        is_series = True
        subplots = False
    elif not features:
        features = data_df.columns

    if subplots:
        return _subplot_generator(data_df, name, features, layout, 'plot')

    else:
        for feature in features:
            plt.clf()
            if is_series:
                time_serie = data_df
            else:
                time_serie = data_df[feature].copy()
            time_serie.plot(figsize=(15, 10))
            if save:
                plt.savefig(Path(plot_location, f"{name}_{feature}.png"), dpi=200)

    return plt.gcf()


def acorr(data_df, name="", features=None, subplots=True, layout=(27, 2), maxlags=100, save=False):
    """
    Autocorrelation plots.

    Parameters:
        data_df:    data frame containing the data which is to be plotted.
        name:       name under which to save the plot
        features:   columns from the frame which are to be plotted
        layout:     of the plot in rows x columns
    
    Returns:
        fig:    matplotlib generated figure obj
        axs:    obj 
    """

    plt.clf()

    if subplots:
        if not features:
            features = data_df.columns
        return _subplot_generator(data_df, name, features, layout, 'acorr')
    else:
        if not features:
            features = data_df.columns

        for feature in features:
            plt.acorr(data_df[feature], maxlags=maxlags)

            if save:
                plt.savefig(Path(plot_location, f"{feature}.png"), dpi=200)

            plt.clf()

        return None


def hist(data_df, name="", features=None, subplots=True, layout=(27, 2), save=False):
    """
    Histogram plots.

    Parameters:
        data_df:    data frame containing the data which is to be plotted.
        name:       name under which to save the plot
        features:   columns from the frame which are to be plotted
        layout:     of the plot in rows x columns
    
    Returns:
        fig:    matplotlib generated figure obj
        axs:    obj 
    """
    plt.clf()
    if not features:
        features = data_df.columns
    if subplots:
        return _subplot_generator(data_df, name, features, layout, 'hist')
    else:
        if not features:
            features = data_df.columns

        for feature in features:
            plt.hist(data_df[feature])
            if save:
                plt.savefig(Path(plot_location, f"{feature}.png"), dpi=200)
            plt.clf()

        return None


def corrMatrix(data_df, features=None):
    """
    Correlation plots.

    Parameters:
        data_df:    data frame containing the data which is to be plotted.
        features:   columns from the frame which are to be plotted
    
    Returns:
        plt:    matplotlib generated pyplot obj
    """
    plt.clf()
    if not features:
        features = data_df.columns

    data = data_df[features]
    corrMatrix = data.corr()
    size = (np.max([0.4 * len(features), 5]), np.max([0.4 * len(features), 5]))
    fig, axs = plt.subplots(1, 1, figsize=size)
    axs = sn.heatmap(corrMatrix, annot=False)
    plt.subplots_adjust(bottom=0.35)
    plt.subplots_adjust(left=0.45)
    plt.savefig(Path(plot_location, "corrMatrix.png"), dpi=200)
    return plt.gcf()


def make_sample_plot(model, generator, folder=None, batches=20, title="", bin_averages=None):
    """
    """
    y_pred, y_true = make_prediction_vector(model=model,
                                            generator=generator,
                                            batches=batches,
                                            bin_averages=bin_averages)

    fig, ax = plt.subplots()
    pd.DataFrame(y_true, columns=['y_true']).plot(ax=ax, ylabel="load")
    pd.DataFrame(y_pred.reshape(-1, 1), columns=['y_pred']).plot(color="r", ax=ax)

    if title:
        plt.title(title)

    if folder:
        plt.savefig(Path(folder, f"{title}_sample.png"))

    return fig


def make_sample_subplot(model,
                        generators,
                        titles=None,
                        folder=None,
                        batches=20,
                        title="",
                        bin_averages=None,
                        layout=None):
    """
    """
    if not isinstance(generators, list):
        generators = [generators]

    if not layout:
        layout = (len(generators), 1)

    if not titles:
        titles = [None] * len(generators)

    fig, ax = plt.subplots(*layout)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

    for index, (generator, title) in enumerate(zip(generators, titles)):
        cur_ax = get_ax(ax, index, layout)

        y_pred, y_true = make_prediction_vector(model=model,
                                                generator=generator,
                                                batches=batches,
                                                bin_averages=bin_averages)

        pd.DataFrame(y_true.reshape(-1, 1), columns=['y_true']).plot(ax=cur_ax, ylabel="load")
        pd.DataFrame(y_pred.reshape(-1, 1), columns=['y_pred']).plot(color="r", ax=cur_ax)

        if title:
            cur_ax.set_title(title)

    if folder:
        plt.savefig(Path(folder, f"{title}_sample.png"))

    return fig


def get_ax(ax, index, layout):
    """_summary_

    Args:
        ax (_type_): _description_
        index (_type_): _description_
        layout (_type_): _description_
    """
    if layout[1] == 1:
        rows = int(np.floor(index / layout[1]))
        return ax[rows]
    else:
        rows = int(np.floor(index / layout[1]))
        cols = index % layout[1]
        return ax[rows, cols]


def make_history_plot(history, folder=None, title=None, train_key="loss", val_key="val_loss"):
    """
    """
    fig, ax = plt.subplots()
    pd.DataFrame(history[train_key], columns=[train_key]).plot(ax=ax, ylabel="y_true")
    if val_key in history.keys():
        pd.DataFrame(history[val_key], columns=[val_key]).plot(color="r", ax=ax, ylabel="y_pred")

    if title:
        plt.title(title)

    if folder:
        plt.savefig(Path(folder, "loss.png"))

    return plt


def make_history_plots(history, train_keys, val_keys, folder=None, titles=None):
    """
    """
    layout = (len(train_keys), 1)

    fig, ax = plt.subplots(*layout)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

    if not titles:
        titles = [None] * len(train_keys)

    for index, (train_key, val_key, title) in enumerate(zip(train_keys, val_keys, titles)):
        cur_ax = get_ax(ax, index, layout)
        pd.DataFrame(history[train_key], columns=[train_key]).plot(ax=cur_ax, ylabel="y_true")
        pd.DataFrame(history[val_key], columns=[val_key]).plot(color="r",
                                                               ax=cur_ax,
                                                               ylabel="y_pred")

        if title:
            cur_ax.set_title(title)

    if folder:
        plt.savefig(Path(folder, "loss.png"))

    return fig
