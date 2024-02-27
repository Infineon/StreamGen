"""🧪 `streamgen.samplers.tree.SamplingTree` tests."""
# ruff: noqa: S101, D103, ANN001, ANN201, PLR2004

from typing import Any

import numpy as np
import pytest

from streamgen.parameter import Parameter
from streamgen.parameter.store import ParameterStore
from streamgen.samplers.tree import SamplingTree
from streamgen.transforms import BranchingNode, TransformNode, noop

# ---------------------------------------------------------------------------- #
# *                             helper functions                               #
# ---------------------------------------------------------------------------- #


def noise(input_, params: dict[str, Any]):  # noqa: ARG001
    return np.random.randn(params["size"],params["size"])  # noqa: NPY002

def skew(input, params: dict[str, Any]):  # noqa: A002
    return input * np.linspace(params["skew"], 1.0, input.shape[0])

def add_random_points(input, params: dict[str, Any]):  # noqa: A002
    for _ in range(params["num_points"]):
        x_ = np.random.randint(input.shape[0]-1)  # noqa: NPY002
        y_ = np.random.randint(input.shape[0]-1)  # noqa: NPY002
        point = np.zeros_like(input)
        point[x_,y_] = 8.0
        input += point  # noqa: A001
    return input


# ---------------------------------------------------------------------------- #
# *                                 fixtures                                   #
# ---------------------------------------------------------------------------- #




# ---------------------------------------------------------------------------- #
# *                                   tests                                    #
# ---------------------------------------------------------------------------- #

def test_sampling_tree():
    """Tests the initialization, sampling and parameter fetching of a `SamplingTree`."""
    tree = SamplingTree(
        [
            TransformNode(noise, Parameter("size", schedule=[16,18], emoji="🦣")),
            TransformNode(skew, Parameter("skew", schedule=[0.0,0.5])),
            BranchingNode(
                [
                    TransformNode(noop),
                    TransformNode(add_random_points, Parameter("num_points", schedule=[1,8])),
                ],
                Parameter("probs", schedule=[[1.0,0.0],[0.0,1.0]]),
            ),
        ],
    )

    sample = tree.sample()

    assert sample.shape == (16,16)

    tree.update()

    sample = tree.sample()

    assert sample.shape == (18,18)
