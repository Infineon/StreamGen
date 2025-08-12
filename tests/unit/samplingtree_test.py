"""🧪 `streamgen.samplers.tree.SamplingTree` tests."""
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


def test_sampling_tree_decision_node_with_probs():
    """Tests the initialization, sampling and parameter fetching of a `SamplingTree`."""
    params = {
        "noise": {
            "size": {
                "schedule": [16, 18],
            },
        },
        "add_random_points": {
            "num_points": {
                "schedule": [1, 8],
            },
        },
    }

    tree = SamplingTree(
        [
            noise,
            TransformNode(skew, Parameter("skew", schedule=[0.0, 0.5]), argument_strategy="dict"),
            {
                "background": [noop, "background"],
                "point pattern": [add_random_points, "points"],
                "probs": Parameter("probs", schedule=[[0.4, 0.6], [0.6, 0.4]]),
            },
        ],
        params,
        collate_func=np.array,
    )

    assert (
        str(tree)
        == """🌳
➡️ `noise(size=16)`
╰── ➡️ `skew(skew=0.0)`
    ╰── 🪴 `branching_node()`
        ├── ➡️ `noop()`
        │   ╰── 🏷️ `background`
        ╰── ➡️ `add_random_points(num_points=1)`
            ╰── 🏷️ `points`
"""
    )

    branching_node = tree.nodes[2]
    assert branching_node.probs.value == [0.4, 0.6]

    sample, label = tree.sample()

    assert sample.shape == (16, 16)
    assert label in ["background", "points"]

    tree.update()

    sample, _ = tree.sample()

    assert sample.shape == (18, 18)

    # test iteration
    samples = []
    for idx, (sample, _) in enumerate(tree):
        samples.append(sample)
        if idx == 2:
            break
    assert not np.array_equal(samples[0], samples[1]), "when iterating, the `SamplingTree` should generate different samples."

    samples, labels = tree.collect(64)
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (64, 18, 18)
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (64,)


def test_sampling_tree_decision_node_without_probs():
    """Tests the initialization, sampling and parameter fetching of a `SamplingTree`."""
    params = ParameterStore(
        {
            "noise": {
                "size": {
                    "schedule": [16, 18],
                },
            },
            "add_random_points": {
                "num_points": {
                    "schedule": [1, 8],
                },
            },
            "branching_node": {
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
        ],
        params,
    )

    branching_node = tree.nodes[1]
    assert branching_node.probs.value == [1.0, 0.0], "Probs should be fetched from params since there is a scope `decision`."

    (sample, target) = tree.sample()

    assert sample.shape == (16, 16)
    assert target == "background"

    samples = tree.collect(10, "stochastic")
    assert len(samples) == 10
    targets = {target for sample, target in samples}
    assert targets == {"background"}

    samples = tree.collect(10, "balanced")
    assert len(samples) == 30
    targets = {target for _sample, target in samples}
    assert targets == {"background", "points", "skewed points"}

    samples = tree.collect(10, "balanced pruned")
    assert len(samples) == 10, "since only the background branch has probs > 0"
    targets = {target for _sample, target in samples}
    assert targets == {"background"}

    tree.set_update_step(1)

    assert branching_node.probs.value == [0.0, 1.0]

    (sample, target) = tree.sample()

    assert sample.shape == (18, 18)
    assert target in ["points", "skewed points"], "not sure, since we have two paths with probs > 0"

    samples = tree.collect(10, "balanced pruned")
    assert len(samples) == 20
    targets = {target for _sample, target in samples}
    assert targets == {"points", "skewed points"}

    tree.set_update_step(0)

    branching_node = tree.nodes[1]
    assert branching_node.probs.value == [1.0, 0.0]

    (sample, target) = tree.sample()

    assert sample.shape == (16, 16)

    # test get_params
    tree.set_update_step(0)
    tree_params = tree.get_params()
    assert all(param in tree_params.parameter_names for param in params.parameter_names)
    assert all(tree_params[param].value == params[param].value for param in params.parameter_names)


def test_merging_after_branching():
    """🪴🔀 tests the merging of branches."""
    tree = SamplingTree(
        [
            lambda input: 0,  # noqa: ARG005
            {
                "probs": Parameter("probs", schedule=[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
                "1": [
                    noop,
                    {
                        "probs": Parameter("probs", schedule=[[1.0, 0.0], [0.0, 1.0]]),
                        "1": [TransformNode(add, Parameter("number", 1)), "one",],
                        "10": [TransformNode(add, Parameter("number", 10)), "two",],
                    },
                ],
                "2": [
                    TransformNode(add, Parameter("number", 2)),
                    "three",
                ],
            },
            noop,
            TransformNode(operate_on_index()(add), Parameter("number", 3)),
            noop,
        ],
    )

    output, target = tree.sample()

    assert output == 4, "The last `partial(add, 3)` transform should be connected to both branches."
    assert target == "one"

    tree.update()
    output, target = tree.sample()

    assert output == 13
    assert target == "two"

    tree.update()
    output, target = tree.sample()

    assert output == 5
    assert target == "three"


def test_tree_visualization(tmp_path):
    """🖌️ tests tree printing and visualization."""
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
            "add_random_points_a_second_time": {
                "num_points": {
                    "value": 4,
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
            "postprocessing_offset": {
                "offset": {
                    "schedule": [0.8, 0.6],
                },
            },
        },
    )
    tree = SamplingTree(
        [
            noise,
            {
                "name": "decision",
                "background": [noop, ClassLabelNode("no-pattern")],
                "patterns": [
                    add_random_points,
                    {
                        "name": "skew_decision",
                        "noop": [noop, ClassLabelNode("points")],
                        "skew": [
                            TransformNode(skew, Parameter("skew", schedule=[0.0, 0.5]), argument_strategy="dict", emoji="🔩"),
                            TransformNode(add_random_points, name="add_random_points_a_second_time"),
                            ClassLabelNode("skewed points"),
                        ],
                    },
                ],
            },
            TransformNode(lambda input, offset: input + offset, name="postprocessing_offset", emoji="➕"),
        ],
        params,
    )

    assert (
        str(tree)
        == """🌳
➡️ `noise(size=16)`
╰── 🪴 `decision()`
    ├── ➡️ `noop()`
    │   ╰── 🏷️ `no-pattern`
    │       ╰── ➕ `postprocessing_offset(offset=0.8)`
    ╰── ➡️ `add_random_points(num_points=1)`
        ╰── 🪴 `skew_decision()`
            ├── ➡️ `noop()`
            │   ╰── 🏷️ `points`
            │       ╰── ➕ `postprocessing_offset(offset=0.8)`
            ╰── 🔩 `skew(skew=0.0)`
                ╰── ➡️ `add_random_points_a_second_time(num_points=4)`
                    ╰── 🏷️ `skewed points`
                        ╰── ➕ `postprocessing_offset(offset=0.8)`
"""
    )

    tree.to_dotfile(tmp_path / "tree.dot")

    tree.to_svg(tmp_path / "tree")


def test_get_paths():
    """🍃 tests the extraction of deterministic paths to each leaf."""
    params = pd.DataFrame(
        {
            "add.number": [1, 2],
            "branching_node.probs": [[0.6, 0.4], [0.4, 0.6]],
        },
    )

    tree = SamplingTree(
        [
            lambda input: 0,  # noqa: ARG005
            {
                "class": [
                    add,
                    "add",
                ],
                "background": [
                    "noop",
                ],
            },
        ],
        params,
    )

    paths = tree.get_paths()

    assert len(paths) == 2
    assert len(paths[0].nodes) == 3
    assert len(paths[1].nodes) == 2

    output, target = paths[0].sample()
    assert target == "add"
    assert output == 1

    output, target = paths[1].sample()
    assert target == "noop"
    assert output == 0
