"""🧪 `streamgen.streams` tests."""

# ruff: noqa: S101, D103, ANN001, ANN201, PLR2004, ERA001
from typing import Any

import numpy as np
import pytest

# from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from streamgen.nodes import TransformNode
from streamgen.parameter import Parameter
from streamgen.parameter.store import ParameterStore
from streamgen.samplers.tree import SamplingTree
from streamgen.streams import collect_stream, construct_continuum_scenario  # , construct_avalanche_classification_datasets
from streamgen.transforms import LabelEncoder, noop, operate_on_index

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


# ---------------------------------------------------------------------------- #
# *                                 fixtures                                   #
# ---------------------------------------------------------------------------- #


@pytest.fixture
def tree() -> SamplingTree:
    """A simple sampling tree that generates labeled time series."""
    params = ParameterStore(
        {
            "noise": {
                "size": {
                    "schedule": [16, 18],
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
                    "schedule": [[0.3, 0.7], [0.9, 0.1]],
                },
            },
            "skew_decision": {
                "probs": {
                    "schedule": [[1.0, 0.0], [0.0, 1.0]],
                },
            },
        },
    )
    return SamplingTree(
        [
            noise,
            {
                "name": "decision",
                "background": [noop, "background"],
                "patterns": [
                    add_random_points,
                    {
                        "name": "skew_decision",
                        "noop": [noop, "points"],
                        "skew": [
                            TransformNode(skew, Parameter("skew", schedule=[0.0, 0.5]), argument_strategy="dict"),
                            add_random_points,
                            "skewed points",
                        ],
                    },
                ],
            },
            operate_on_index(1)(LabelEncoder(["background", "points", "skewed points"])),
        ],
        params,
        collate_func=np.array,
    )


# ---------------------------------------------------------------------------- #
# *                                   tests                                    #
# ---------------------------------------------------------------------------- #


def test_construct_stream(tree) -> None:
    """Tests the construction of a stream."""
    stream = collect_stream(tree, 2, 250)

    assert len(stream) == 2

    first_experience = stream[0]
    samples, labels = first_experience

    assert len(samples) == 250
    assert samples.shape == (250, 16, 16)
    assert labels.shape == (250,)


# def test_avalanche_stream(tree) -> None:
#     """Tests the construction of an avalanche benchmark."""
#     experiences = collect_stream(tree, 2, 250)

#     stream = construct_avalanche_classification_datasets(experiences)

#     assert len(stream) == 2

#     benchmark = benchmark_from_datasets(
#         train=stream,
#     )

#     assert len(benchmark.train_stream) == 2

#     exp = benchmark.train_stream[0]
#     # current experience is the position of the experience in the stream.
#     # It must never be used during training or evaluation
#     # if you try to use it will fail with a MaskedAttributeError

#     assert exp.current_experience == 0


def test_continuum_stream(tree) -> None:
    """Tests the construction of a continuum scenario."""
    experiences = collect_stream(tree, 2, 250)

    scenario = construct_continuum_scenario(experiences)

    assert len(scenario) == 2

    task_id, train_taskset = next(enumerate(scenario))

    assert task_id == 0
    assert len(train_taskset) == 250
