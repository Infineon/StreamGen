"""🧪 `streamgen.samplers.tree.SamplingTree` reproducibility tests."""
# ruff: noqa: S101, D103, ANN001, ANN201, PLR2004

from typing import Any

import numpy as np
import pandas as pd

from streamgen.nodes import ClassLabelNode, TransformNode
from streamgen.parameter import Parameter
from streamgen.parameter.store import ParameterStore
from streamgen.samplers.tree import SamplingTree
from streamgen.transforms import noop, operate_on_index

# ---------------------------------------------------------------------------- #
# *                             helper functions                               #
# ---------------------------------------------------------------------------- #


def noise(input_, size):  # noqa: ARG001
    return np.random.randn(size, size)  # noqa: NPY002


def skew(input, params: dict[str, Any]):  # noqa: A002
    return input * np.linspace(params["skew"].value, 1.0, input.shape[0])


def add_random_points(input, num_points):  # noqa: A002
    for _ in range(num_points):
        x_ = np.random.randint(input.shape[0] - 1)  # noqa: NPY002
        y_ = np.random.randint(input.shape[0] - 1)  # noqa: NPY002
        point = np.zeros_like(input)
        point[x_, y_] = 8.0
        input += point  # noqa: A001
    return input


def add(input: int, number):  # noqa: A002
    return input + number

def add_and_subtract(input: int, number, number2): # noqa: A002
    return input + number - number2


# ---------------------------------------------------------------------------- #
# *                                 fixtures                                   #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# *                                   tests                                    #
# ---------------------------------------------------------------------------- #

def test_parameter_fetching_from_global_scope():
    """Tests if nodes fetch their missing arguments from the top-level/global scope."""
    tree = SamplingTree(
        [
            lambda input: 0,  # noqa: ARG005
            {
                "probs": Parameter("probs", schedule=[[1.0, 0.0], [0.0, 1.0]]),
                "1": [
                    add,
                    "one",
                ],
                "2": [
                    TransformNode(add, name="two"),
                    add_and_subtract,
                    "two",
                ],
            },
            TransformNode(operate_on_index()(add), Parameter("number", 3)),
        ],
        {
            "two": {
                "number": 3
            },
            "add_and_subtract": {
                "number": 5
            },
            "number": 1,
            "number2": 2
        }
    )

    output, target = tree.sample()

    assert output == 4, "The last `partial(add, 3)` transform should be connected to both branches."
    assert target == "one"

    tree.update()
    output, target = tree.sample()

    assert output == 9
    assert target == "two"
