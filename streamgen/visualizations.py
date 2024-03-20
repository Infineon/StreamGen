"""🖼️parameter visualization functions."""

from copy import deepcopy
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from streamgen.parameter import Parameter


def plot_parameter(param: Parameter, num_values: int | None = None, ax: plt.Axes | None = None) -> plt.Axes:
    """📈 plots the scheduled values of a parameter.

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

    ax.set_title(param.name)
    return ax
