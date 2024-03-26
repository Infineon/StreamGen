"""🧪 `streamgen.visualizations` tests."""
# ruff: noqa: S101, D103, ANN001, ANN201, PLR2004
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from streamgen import visualizations
from streamgen.parameter import Parameter
from streamgen.parameter.store import ParameterStore


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


def test_store_plotting() -> None:
    """Tests the plotting of a parameter store."""
    df = pd.DataFrame(
        {
            "background.signal_length": 256,
            "background.offset": 0.0,
            "background.strength": 0.1,
            "branching point.probs": [
                [0.5, 0.5, 0.0],
                [0.1, 0.9, 0.0],
                [0.0, 0.1, 0.9],
            ],
            "ramp.height": 1.0,
            "ramp.length": 128,
            "step.length": 128,
            "step.kernel_size": 1,
        },
    )
    store = ParameterStore.from_dataframe(df)

    fig = visualizations.plot_parameter_store(store, num_values=3)
