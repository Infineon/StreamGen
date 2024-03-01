"""🧪 `streamgen.samplers.tree.SamplingTree` tests."""
# ruff: noqa: S101, D103, ANN001, ANN201, PLR2004

from typing import Any

import numpy as np

from streamgen.nodes import ClassLabelNode, TransformNode
from streamgen.parameter import Parameter
from streamgen.parameter.store import ParameterStore
from streamgen.samplers.tree import SamplingTree
from streamgen.transforms import noop, operate_on_key

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


@operate_on_key("input")
def add(input: int, number):  # noqa: A002
    return input + number


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

    assert (
        str(tree)
        == """🌳
➡️ `noise(size=16)`
╰── ➡️ `skew(skew=0.0)`
    ╰── 🪴 `branching point()`
        ├── ➡️ `noop()`
        ╰── ➡️ `add_random_points(num_points=1)`
"""
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


def test_merging_after_branching():
    """🪴🔀 tests the merging of branches."""
    tree = SamplingTree(
        [
            lambda input: {"input": 0, "target": None},  # noqa: A002, ARG005
            {
                "probs": Parameter("probs", schedule=[[1.0, 0.0], [0.0, 1.0]]),
                "1": [
                    TransformNode(add, Parameter("number", 1)),
                    ClassLabelNode("one"),
                ],
                "2": [
                    TransformNode(add, Parameter("number", 2)),
                    ClassLabelNode("two"),
                ],
            },
            TransformNode(add, Parameter("number", 3)),
        ],
    )

    sample = tree.sample()

    assert sample["input"] == 4, "The last `partial(add, 3)` transform should be connected to both branches."
    assert sample["target"] == "one"

    tree.update()
    sample = tree.sample()

    assert sample["input"] == 5
    assert sample["target"] == "two"


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
            TransformNode(lambda input: input + 0.8, name="postprocessing_offset", emoji="➕"),  # noqa: RUF001, A002
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
    │       ╰── ➕ `postprocessing_offset()`
    ╰── ➡️ `add_random_points(num_points=1)`
        ╰── 🪴 `skew_decision()`
            ├── ➡️ `noop()`
            │   ╰── 🏷️ `points`
            │       ╰── ➕ `postprocessing_offset()`
            ╰── 🔩 `skew(skew=0.0)`
                ╰── ➡️ `add_random_points_a_second_time(num_points=4)`
                    ╰── 🏷️ `skewed points`
                        ╰── ➕ `postprocessing_offset()`
"""
    )

    tree.to_dotfile(tmp_path / "tree.dot")

    with (tmp_path / "tree.dot").open(encoding="utf8") as f:
        assert (
            f.read()
            == """digraph tree {
    "0x0" [label="➡️ `noise(size=16)`"];
    "0x1" [label="🪴 `decision()`",shape=diamond];
    "0x2" [label="➡️ `noop()`"];
    "0x3" [label="🏷️ `no-pattern`",shape=cds];
    "0x4" [label="➕ `postprocessing_offset()`"];
    "0x5" [label="➡️ `add_random_points(num_points=1)`"];
    "0x6" [label="🪴 `skew_decision()`",shape=diamond];
    "0x7" [label="➡️ `noop()`"];
    "0x8" [label="🏷️ `points`",shape=cds];
    "0x9" [label="➕ `postprocessing_offset()`"];
    "0xa" [label="🔩 `skew(skew=0.0)`"];
    "0xb" [label="➡️ `add_random_points_a_second_time(num_points=4)`"];
    "0xc" [label="🏷️ `skewed points`",shape=cds];
    "0xd" [label="➕ `postprocessing_offset()`"];
    "0x0" -> "0x1";
    "0x1" -> "0x2";
    "0x1" -> "0x5";
    "0x2" -> "0x3";
    "0x3" -> "0x4";
    "0x5" -> "0x6";
    "0x6" -> "0x7";
    "0x6" -> "0xa";
    "0x7" -> "0x8";
    "0x8" -> "0x9";
    "0xa" -> "0xb";
    "0xb" -> "0xc";
    "0xc" -> "0xd";
}
"""
        )
