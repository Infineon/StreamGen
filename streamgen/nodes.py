"""🪢 different node implementations using [anytree](https://anytree.readthedocs.io/en/stable/) `NodeMixin`."""

from collections.abc import Callable
from typing import Any, Protocol

import anytree
import numpy as np
from beartype import beartype
from loguru import logger

from streamgen.enums import ArgumentPassingStrategy, ArgumentPassingStrategyLit
from streamgen.parameter import Parameter
from streamgen.parameter.store import ParameterStore
from streamgen.transforms import noop


class Traverse(Protocol):
    """🏃 transform-node traversal protocol `(input: Any) -> (output, anytree.NodeMixin | None)`.

    If a node has children, return the next node to traverse. Otherwise return None and stop traversal.
    """

    def traverse(input: Any) -> tuple[Any, anytree.NodeMixin | None]:  # noqa: D102, N805, A002
        ...


class TransformNode(anytree.NodeMixin):
    """🪢 parametric transform node base class using [anytree](https://anytree.readthedocs.io/en/stable/) `NodeMixin`.

    A node can be linked dynamically to other nodes via the `parent` and `children` attributes.
    The main idea of a `TransformNode` is that is contains a transformation, which is called when the node is traversed.

    The parameters of the node can be updated.

    Args:
        transform (Callable): any callable function
        params (Parameter | ParameterStore | None, optional): parameters/arguments for the `transform`.
            Can by fetched dynamically from a `ParameterStore` using `TransformNode.fetch_params`. Defaults to None.
        argument_strategy (ArgumentPassingStrategy | ArgumentPassingStrategyLit, optional):
            strategy which defines how parameters are passed to functions. Defaults to "unpack", where params are passed as kwargs.
        name (str | None, optional): node name. If none, use `transform.__name__`. Defaults to None.
        emoji (str, optional): emoji for string representation. Defaults to "➡️".
    """

    @beartype()
    def __init__(  # noqa: D107, PLR0913
        self,
        transform: Callable,
        params: Parameter | ParameterStore | None = None,
        argument_strategy: ArgumentPassingStrategy | ArgumentPassingStrategyLit = "unpack",
        name: str | None = None,
        emoji: str = "➡️",
    ) -> None:
        super().__init__()
        self.transform = transform
        self.params: ParameterStore | None = ParameterStore([params]) if isinstance(params, Parameter) else params
        self.argument_strategy = argument_strategy

        self.name = name if name else transform.__name__
        self.emoji = emoji
        self.parent = None

    def traverse(self, input: Any) -> tuple[Any, anytree.NodeMixin | None]:  # noqa: A002, ANN401
        """🏃 `streamgen.transforms.Traverse` protocol `(input: Any) -> (output, anytree.NodeMixin | None)`.

        If `self` has children, return the next node to traverse. Otherwise return None and stop traversal.

        Args:
            input (Any): any input

        Returns:
            tuple[Any, anytree.NodeMixin | None]: output and potential next node to traverse
        """
        match (self.params, self.argument_strategy):
            case (None, _):
                output = self.transform(input)
            case (_, ArgumentPassingStrategy.DICT):
                output = self.transform(input, self.params)
            case (_, ArgumentPassingStrategy.UNPACK):
                output = self.transform(input, **self.params.get_params())

        if self.children == ():
            return output, None

        if len(self.children) != 1:
            logger.warning(f"Node {self} contains more than one child. Only the first one is traversed.")

        return output, self.children[0]

    def update(self) -> None:
        """🆙 updates every parameter."""
        if self.params:
            self.params.update()

    def fetch_params(self, params: ParameterStore) -> None:
        """⚙️ fetches params from a ParameterStore.

        If the node was explicitly parameterized, use those params.

        Args:
            params (ParameterStore): _description_
        """
        if self.params:
            return
        if self.name in params.scopes:
            self.params = params.get_scope(self.name)

    def __str__(self) -> str:
        """🏷️ Returns the string representation `str(self)`.

        Returns:
            str: string representation of self
        """
        s = f"{self.emoji} `{self.name}`"

        if self.params:
            s += f" with {self.params}"

        return s


class BranchingNode(TransformNode):
    """🪴 node with multiple children/branches.

    When traversed, a random branch is selected based on the probabilities defined by `probs`.

    Args:
        branches (dict): dictionary, where each key:value pair represent label:branch.
        probs (Parameter | None, optional): parameter containing the probabilities for selecting each branch.
            `probs.value` is passed to `numpy.random.choice` as parameter `p`, which is documented as:
            (1-D array_like, optional) the probabilities associated with each entry in a.
            If not given the sample assumes a uniform distribution over all entries. Defaults to None.
        name (str | None, optional): name of the node. Important for fetching the `probs` if not present. Defaults to "branching point".
        seed (int, optional): random number generator seed. Defaults to 42.
    """

    def __init__(  # noqa: D107
        self,
        branches: dict,
        probs: Parameter | None = None,
        name: str | None = None,
        seed: int = 42,
    ) -> None:
        self.name = name if name else "branching point"
        self.probs = probs
        self.rng = np.random.default_rng(seed)

        self.branches = {branch_name: construct_graph(nodes) for branch_name, nodes in branches.items()}

        self.children = [branch[0] for branch in self.branches.values()]

        super().__init__(transform=noop, name=self.name, emoji="🪴")

    def traverse(self, input: Any) -> tuple[Any, anytree.NodeMixin]:  # noqa: A002, ANN401
        """🏃🎲 `streamgen.transforms.Traverse` protocol `(input: Any) -> (output, anytree.NodeMixin | None)`.

        During traversal, a branching node samples the next node from its children.

        Args:
            input (Any): any input

        Returns:
            tuple[Any, anytree.NodeMixin | None]: output and next node to traverse
        """
        key = self.rng.choice(list(self.branches.keys()), p=self.probs.value if self.probs else None)
        next_node = self.branches[key][0]

        return input, next_node

    def update(self) -> None:
        """🆙 updates every parameter."""
        if self.probs:
            self.probs.update()

        for branch in self.branches.values():
            for node in branch:
                node.update()

    def fetch_params(self, params: ParameterStore) -> None:
        """⚙️ fetches params from a ParameterStore.

        Args:
            params (ParameterStore): parameter store to fetch the params from
        """
        if self.probs is None and self.name in params.scopes:
            probs = list(params.get_scope(self.name).parameters.values())
            assert (  # noqa: S101
                len(probs) == 1
            ), f'Make sure to only have a single parameter in the scope "{self.name}" when setting the parameters of a `BranchingNode` through `fetch_params`.'  # noqa: E501, S101
            self.probs = probs[0]

        for branch in self.branches.values():
            for node in branch:
                node.fetch_params(params)


@beartype()
def construct_graph(nodes: Callable | TransformNode | dict | list[Callable | TransformNode | dict]) -> list[TransformNode]:
    """🏗️ assembles and links nodes into a graph/tree.

    The following rules apply during construction:

    1. Nodes are linked sequentially according to the ordering in the top-level list.
    2. `TransformNode` and sub-classes are not modified.
    3. `Callable`s are cast into `TransformNode`s.
    4. dictionaries are interpreted as `BranchingNode`s, where each value represents a branch.
        The keys `name`, `probs` and `seed` are reserved to describe the node itself.

    Args:
        nodes (Callable | TransformNode | dict | list[Callable | TransformNode | dict]): pythonic short-hand description of a graph/tree

    Returns:
        list[TransformNode]: list of linked nodes
    """
    # We need the next two lines to handle single element branches gracefully in the recursion.
    if not isinstance(nodes, list):
        nodes = [nodes]

    graph = []
    for node in nodes:
        match node:
            case Callable():
                graph.append(TransformNode(node))
            case TransformNode():
                graph.append(node)
            case dict():
                name = node.pop("name", None)
                probs = node.pop("probs", None)
                seed = node.pop("seed", 42)
                graph.append(BranchingNode(node, name=name, probs=probs, seed=seed))

    # connect the nodes to enable traversal
    for idx, node in enumerate(graph[:-1]):
        node.children = [graph[idx + 1]]

    return graph
