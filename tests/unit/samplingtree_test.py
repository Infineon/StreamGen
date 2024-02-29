"""🧪 `streamgen.samplers.tree.SamplingTree` tests."""
# ruff: noqa: S101, D103, ANN001, ANN201, PLR2004

from typing import Any

import numpy as np

from streamgen.nodes import BranchingNode, TransformNode
from streamgen.parameter import Parameter
from streamgen.parameter.store import ParameterStore
from streamgen.samplers.tree import SamplingTree
from streamgen.transforms import noop

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


# ---------------------------------------------------------------------------- #
# *                                 fixtures                                   #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# *                                   tests                                    #
# ---------------------------------------------------------------------------- #


def test_sampling_tree_decision_node_with_probs():
    """Tests the initialization, sampling and parameter fetching of a `SamplingTree`."""
    params = ParameterStore(
        {
            "noise": {
                "size": {
                    "schedule": [16, 18],
                    "emoji": "🦣",
                },
            },
            "add_random_points": {
                "num_points": {
                    "schedule": [1, 8],
                },
            },
        },
    )
    tree = SamplingTree(
        [
            noise,
            TransformNode(skew, Parameter("skew", schedule=[0.0, 0.5]), argument_strategy="dict"),
            {
                "background": noop,
                "point pattern": add_random_points,
                "probs": Parameter("probs", schedule=[[0.4, 0.6], [0.6, 0.4]]),
            },
        ],
        params,
    )

    branching_node = tree.nodes[2]
    assert branching_node.probs.value == [0.4, 0.6]

    sample = tree.sample()

    assert sample.shape == (16, 16)

    tree.update()

    sample = tree.sample()

    assert sample.shape == (18, 18)


def test_sampling_tree_decision_node_without_probs():
    """Tests the initialization, sampling and parameter fetching of a `SamplingTree`."""
    params = ParameterStore(
        {
            "noise": {
                "size": {
                    "schedule": [16, 18],
                    "emoji": "🦣",
                },
            },
            "add_random_points": {
                "num_points": {
                    "schedule": [1, 8],
                },
            },
            "branching point": {
                "probs": {
                    "schedule": [[1.0, 0.0], [0.0, 1.0]],
                },
            },
        },
    )
    tree = SamplingTree(
        [
            noise,
            TransformNode(skew, Parameter("skew", schedule=[0.0, 0.5]), argument_strategy="dict"),
            {
                "background": noop,
                "point pattern": add_random_points,
            },
        ],
        params,
    )

    branching_node = tree.nodes[2]
    assert branching_node.probs.value == [1.0, 0.0], "Probs should be fetched from params since there is a scope `branching point`."

    sample = tree.sample()

    assert sample.shape == (16, 16)

    tree.update()

    assert branching_node.probs.value == [0.0, 1.0]

    sample = tree.sample()

    assert sample.shape == (18, 18)


def test_sampling_tree_deep_nesting():
    """Tests the initialization, sampling and parameter fetching of a `SamplingTree`."""
    params = ParameterStore(
        {
            "noise": {
                "size": {
                    "schedule": [16, 18],
                    "emoji": "🦣",
                },
            },
            "skew": {
                "skew": {
                    "schedule": [0.0, 0.5],
                },
            },
            "add_random_points": {
                "num_points": {
                    "schedule": [1, 8],
                },
            },
            "decision": {
                "probs": {
                    "schedule": [[1.0, 0.0], [0.0, 1.0]],
                },
            },
            "skew_decision": {
                "probs": {
                    "schedule": [[0.3, 0.7], [0.9, 0.1]],
                },
            },
        },
    )
    tree = SamplingTree(
        [
            noise,
            {
                "name": "decision",
                "background": noop,
                "patterns": [
                    add_random_points,
                    {
                        "name": "skew_decision",
                        "noop": noop,
                        "skew": [TransformNode(skew, Parameter("skew", schedule=[0.0, 0.5]), argument_strategy="dict"), add_random_points],
                    },
                ],
            },
        ],
        params,
    )

    branching_node = tree.nodes[1]
    assert branching_node.probs.value == [1.0, 0.0], "Probs should be fetched from params since there is a scope `decision`."

    sample = tree.sample()

    assert sample.shape == (16, 16)

    tree.update()

    assert branching_node.probs.value == [0.0, 1.0]

    sample = tree.sample()

    assert sample.shape == (18, 18)
