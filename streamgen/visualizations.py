"""🖼️parameter visualization functions."""

from collections.abc import Callable
from copy import deepcopy
from itertools import islice
from typing import Any

import IPython
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.utils import io
from ipywidgets import widgets
from matplotlib import animation

from streamgen.parameter import Parameter
from streamgen.parameter.store import ParameterStore
from streamgen.samplers.tree import SamplingTree


def plot(values: list[int | float | list | np.ndarray], ax: plt.Axes | None = None, title: str | None = None) -> plt.Axes:
    """📈 plots the scheduled values of a parameter.

    This function currently supports plotting numeric parameters and probabilities.

    Args:
        values (list[int | float | list | np.ndarray]): list of values to plot.
        ax (plt.Axes | None, optional): matplotlib Axes to plot to. Defaults to None.
        title (str | None, optional): title of the plot. Defaults to None

    Raises:
        NotImplementedError: when the type of the parameter values are not yet supported by this function

    Returns:
        plt.Axes: parameter plot
    """
    match values[0]:
        case int() | float():
            ax = sns.lineplot(values, ax=ax)
        case np.ndarray() | list():
            if ax is None:
                fig, ax = plt.subplots()
            sns.set_theme()
            indices = list(range(len(values)))
            values = np.array(values).T
            assert len(values.shape) == 2, "only arrays with two dimensions can be visualized here"
            ax.stackplot(indices, values)
        case _:
            raise NotImplementedError

    ax.set_title(title)
    return ax


def plot_parameter(param: Parameter, num_values: int | None = None, ax: plt.Axes | None = None) -> plt.Axes:
    """⚙️ plots the scheduled values of a parameter.

    This function currently supports plotting numeric parameters and probabilities.

    Args:
        param (Parameter): parameter to plot
        num_values (int | None, optional): number of values to plot.
            If None, collect all values from the schedule. Defaults to None.
        ax (plt.Axes | None, optional): matplotlib Axes to plot to. Defaults to None.

    Raises:
        NotImplementedError: when the type of the parameter values are not yet supported by this function

    Returns:
        plt.Axes: parameter plot
    """
    match num_values:
        case int():
            assert num_values > 1, "at least two value are needed for the plot."
            values = [param.value, *list(islice(deepcopy(param.schedule), num_values - 1))]
        case None:
            values = [param.value, *list(islice(deepcopy(param.schedule), None))]

    return plot(values, ax, title=param.name)


def plot_parameter_store(store: ParameterStore, num_values: int | None = None) -> mpl.figure.Figure:
    """🗄️ plots every parameter in a `ParameterStore` in one figure.

    Args:
        store (ParameterStore): parameter store to plot
        num_values (int | None, optional): number of values to plot.
            If None, collect all values from the schedule. Defaults to None.

    Raises:
        NotImplementedError: when the type of the parameter values are not yet supported

    Returns:
        mpl.figure.Figure: matplotlib figure object
    """
    num_columns: int = len(store.parameter_names)

    sns.set_theme()
    fig = plt.figure()

    for idx, param in enumerate(store.parameter_names):
        ax = fig.add_subplot(num_columns, 1, idx + 1)
        plot_parameter(store[param], num_values=num_values, ax=ax)

    fig.set_figheight(num_columns * 3.0)
    plt.tight_layout()

    return fig


def plot_parameter_store_widget(store: ParameterStore, num_values: int | None = None) -> widgets.Tab:
    """📂 plots every parameter of each scope in a `ParameterStore` in a separate `ipywidgets.widgets.Tab`.

    Args:
        store (ParameterStore): parameter store to plot
        num_values (int | None, optional): number of values to plot.
            If None, collect all values from the schedule. Defaults to None.

    Raises:
        NotImplementedError: when the type of the parameter values are not yet supported

    Returns:
        widgets.Tab: ipywidgets tab widget
    """
    scopes = list(store.scopes)
    tabs = [widgets.Output() for _ in scopes]
    widget = widgets.Tab(children=tabs)

    for idx, scope in enumerate(scopes):
        widget.set_title(idx, scope)
        with tabs[idx]:
            params = store.get_scope(scope)
            fig = plot_parameter_store(params, num_values=num_values)
            plt.show(fig)

    return widget


def plot_labeled_samples_grid(
    tree: SamplingTree,
    plotting_func: Callable[[Any, plt.Axes], plt.Axes],
    columns: int = 4,
) -> mpl.figure.Figure:
    """📟 plots a `columns`x`columns` grid of labeled samples generated from a `SamplingTree` with `ClassLabelNode`s.

    Args:
        tree (SamplingTree): tree to generate samples from
        plotting_func (Callable[[Any, plt.Axes], plt.Axes]): function to visualize a single sample.
            The function should take a sample and a `plt.Axes` as arguments.
        columns (int, optional): number of samples in the columns (and rows). Defaults to 4.

    Returns:
        mpl.figure.Figure: matplotlib figure object
    """
    num_samples = columns * columns
    with io.capture_output() as _captured:
        samples = tree.collect(num_samples)

    sns.set_theme()
    fig = plt.figure()

    for idx, (sample, target) in enumerate(samples):
        ax = fig.add_subplot(columns, columns, idx + 1)
        plotting_func(sample, ax)
        ax.set_title(target)

    fig.set_figheight(columns * 3.0)
    fig.set_figwidth(columns * 3.0)
    plt.tight_layout()

    return fig


def plot_labeled_samples_animation(
    tree: SamplingTree,
    plotting_func: Callable[[Any, plt.Axes], plt.Axes],
    num_samples: int = 8,
    interval: int = 200,
) -> IPython.display.HTML:
    """🎞️ plots several labeled samples generated from a `SamplingTree` with `ClassLabelNode`s as an animation.

    Args:
        tree (SamplingTree): tree to generate samples from
        plotting_func (Callable[[Any, plt.Axes], plt.Axes]): function to visualize a single sample.
            The function should take a sample and a `plt.Axes` as arguments.
        num_samples (int, optional): number of samples to include in the animation. Defaults to 8.
        interval (int, optional): delay between frames in milliseconds. Defaults to 200.

    Returns:
        IPython.display.HTML: animation player as an ipython HTML output
    """
    with io.capture_output() as _captured:
        samples = tree.collect(num_samples)

        sns.set_theme()
        fig, ax = plt.subplots()

        def _plotting_func(idx: int, ax: plt.Axes) -> None:
            sample, target = samples[idx]
            ax.clear()
            plotting_func(sample, ax)
            ax.set_title(target)

        anim = animation.FuncAnimation(fig, _plotting_func, frames=num_samples, fargs=(ax,), interval=interval)

    return IPython.display.HTML(anim.to_jshtml())
