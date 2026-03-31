"""🧪 regression tests for branching update behavior."""
# ruff: noqa: S101, D103, ANN001, ANN201

from collections import Counter

from streamgen.nodes import TransformNode
from streamgen.parameter import Parameter
from streamgen.samplers.tree import SamplingTree
from streamgen.transforms import noop


# ---------------------------------------------------------------------------- #
# *                             helper functions                               #
# ---------------------------------------------------------------------------- #


def _build_minimal_branch_tree() -> SamplingTree:
    """Builds a small tree where one scoped parameter is attached multiple times.

    Layout:
        root -> BranchingNode(left|right) -> shared_scope

    The node after the branching node is deep-copied into each branch by
    streamgen's tree construction shorthand. All copies fetch params from the
    same scope name ("shared_scope").
    """
    nodes = [
        TransformNode(noop, name="root"),
        {
            "name": "selector",
            "left": [TransformNode(noop, name="left_leaf")],
            "right": [TransformNode(noop, name="right_leaf")],
        },
        TransformNode(noop, name="shared_scope"),
    ]

    params = {
        "shared_scope": {
            "sweep": {
                "schedule": [10, 20, 30],
                "strategy": "hold",
            },
        },
    }

    return SamplingTree(nodes=nodes, params=params, rng=0)


def _collect_schedule_values(mode: str) -> list[int]:
    """Collects two update steps from either tree.update or tree.params.update."""
    tree = _build_minimal_branch_tree()
    parameter = tree.get_params()["shared_scope.sweep"]
    values = [int(parameter.value)]

    for _ in range(2):
        if mode == "tree":
            tree.update()
        elif mode == "params":
            tree.params.update()
        else:
            raise ValueError(f"unknown mode: {mode}")
        values.append(int(parameter.value))

    return values


# ---------------------------------------------------------------------------- #
# *                                   tests                                    #
# ---------------------------------------------------------------------------- #


def test_tree_update_calls_each_parameter_once_in_branching_tree(monkeypatch) -> None:
    """Tests that tree.update touches each parameter object once."""
    tree = _build_minimal_branch_tree()
    target = tree.get_params()["shared_scope.sweep"]

    call_counts: Counter[int] = Counter()
    original_update = Parameter.update

    def wrapped_update(self, *args, **kwargs):
        call_counts[id(self)] += 1
        return original_update(self, *args, **kwargs)

    monkeypatch.setattr(Parameter, "update", wrapped_update)
    tree.update()

    assert call_counts[id(target)] == 1


def test_parameter_store_update_calls_each_parameter_once(monkeypatch) -> None:
    """Tests that ParameterStore.update touches each parameter object once."""
    tree = _build_minimal_branch_tree()
    target = tree.get_params()["shared_scope.sweep"]

    call_counts: Counter[int] = Counter()
    original_update = Parameter.update

    def wrapped_update(self, *args, **kwargs):
        call_counts[id(self)] += 1
        return original_update(self, *args, **kwargs)

    monkeypatch.setattr(Parameter, "update", wrapped_update)
    tree.params.update()

    assert call_counts[id(target)] == 1


def test_tree_update_matches_parameter_store_update_progression() -> None:
    """Tests that tree.update progresses one step and matches params.update."""
    tree_values = _collect_schedule_values("tree")
    params_values = _collect_schedule_values("params")

    assert tree_values == [10, 20, 30]
    assert params_values == [10, 20, 30]
    assert tree_values == params_values
