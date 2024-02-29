"""🧪 `streamgen.nodes` tests."""
# ruff: noqa: S101, D103, ANN001, ANN201, PLR2004

import pytest

from streamgen.nodes import BranchingNode, ClassLabelNode, TransformNode
from streamgen.parameter import Parameter
from streamgen.parameter.store import ParameterStore
from streamgen.samplers.tree import SamplingTree
from streamgen.transforms import operate_on_key

# ---------------------------------------------------------------------------- #
# *                             helper functions                               #
# ---------------------------------------------------------------------------- #

def pure_transform(x):
    return x + 1

def parametric_transform(x, inc):
    return x + inc

def multi_params_transform(x, inc, factor):
    return (x + inc) * factor

@operate_on_key("input")
def add(input: int, number):  # noqa: A002
    return input + number


# ---------------------------------------------------------------------------- #
# *                                 fixtures                                   #
# ---------------------------------------------------------------------------- #


@pytest.fixture()
def multiple_params():
    return ParameterStore(
        Parameter("inc", schedule=[2, 3], emoji="👆"),
        Parameter("factor", schedule=[1.0, 2.0], emoji="🧮"),
    )


@pytest.fixture()
def single_param():
    return Parameter("inc", schedule=[2, 3], emoji="👆")


# ---------------------------------------------------------------------------- #
# *                                   tests                                    #
# ---------------------------------------------------------------------------- #


def test_pure_transform() -> None:
    """💎 tests the behavior of a pure transform node without children."""
    node = TransformNode(pure_transform)

    assert node.name == "pure_transform"
    assert str(node) == "➡️ `pure_transform`"

    out, next_node = node.traverse(0)

    assert next_node is None
    assert out == 1


def test_parametric_transform(single_param):
    """⚙️ tests the behavior of a parametric transform node without children."""
    node = TransformNode(parametric_transform, params=single_param, name="increment", emoji="👆")

    assert node.name == "increment"
    assert str(node) == "👆 `increment` with 🗃️ = {👆 inc: 2}"

    out, next_node = node.traverse(0)

    assert next_node is None
    assert out == 2

    node.update()

    out, next_node = node.traverse(0)

    assert next_node is None
    assert out == 3


def test_connected_nodes(single_param):
    """🔗 tests the traversal behavior of two connected transforms."""
    node1 = TransformNode(pure_transform)
    node2 = TransformNode(parametric_transform, params=single_param, name="increment", emoji="👆")

    # connect node2 to node1
    node2.parent = node1

    out, next_node = node1.traverse(0)

    assert next_node is node2
    assert out == 1

    out, next_node = node2.traverse(out)

    assert next_node is None
    assert out == 3


def test_branching_node(single_param):
    """🪴 tests the behavior of `streamgen.transforms.BranchingNode`."""
    branches = {
        "1": pure_transform,
        "2": [
            TransformNode(parametric_transform, params=single_param),
            {  # nested branching node
                "name": "nested decision",
                "probs": Parameter("probs", value=[0.5, 0.5], emoji="🎲"),
                "seed": 1,
                "1": pure_transform,
                "2": pure_transform,
            },
        ],
    }
    probs = Parameter("probs", value=[1.0, 0.0], emoji="🎲")

    node = BranchingNode(branches, probs)

    assert node.name == "branching point"
    assert str(node) == "🪴 `branching point`"

    out, next_node = node.traverse(0)

    assert out == 0
    assert next_node.name == "pure_transform", "should be `pure_transform`, since its sampling probability is 100%."

def test_class_label_node():
    """🏷️tests the labelling process using `ClassLabelNode`."""
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
        ],
    )

    sample = tree.sample()

    assert sample["input"] == 1
    assert sample["target"] == "one"

    tree.update()
    sample = tree.sample()

    assert sample["input"] == 2
    assert sample["target"] == "two"
