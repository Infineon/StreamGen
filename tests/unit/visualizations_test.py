"""🧪 `streamgen.visualizations` tests."""
# ruff: noqa: S101, D103, ANN001, ANN201, PLR2004
import matplotlib.pyplot as plt
import numpy as np
import pytest

from streamgen import visualizations
from streamgen.parameter import Parameter


def test_parameter_plotting() -> None:
    """Tests the plotting of a parameter."""
    numeric_param = Parameter("param", schedule=np.cos(np.linspace(0.0, np.pi, 100)), strategy="raise exception")
    probs_param_list = Parameter(
        "probs",
        schedule=[
            [0.5, 0.5, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.1, 0.9],
        ],
    )
    probs_param_ndarray = Parameter(
        "probs",
        schedule=np.array(
            [
                [0.5, 0.5, 0.0],
                [0.1, 0.9, 0.0],
                [0.0, 0.1, 0.9],
            ],
        ),
    )
    ndarray_param = Parameter("matrix", schedule=np.ones((3, 3, 3)))
    str_param = Parameter("string", schedule=["a", "b", "c"])

    fig, ax = plt.subplots()
    ax = visualizations.plot_parameter(numeric_param, ax=ax)
    ax = visualizations.plot_parameter(probs_param_list)
    ax = visualizations.plot_parameter(probs_param_ndarray)
    with pytest.raises(AssertionError):
        ax = visualizations.plot_parameter(ndarray_param)
    with pytest.raises(NotImplementedError):
        ax = visualizations.plot_parameter(str_param)
