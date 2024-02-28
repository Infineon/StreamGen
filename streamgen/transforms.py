"""➡️ transform interfaces and helper functions.

Transforms (transformations) are functions with a specific interface.
"""

from collections.abc import Callable
from typing import Any, Protocol

import anytree
import numpy as np
from beartype import beartype
from loguru import logger

from streamgen.parameter import Parameter
from streamgen.parameter.store import ParameterStore

PureTransform = Callable[[Any], Any]
"""💎 any function with the signature `(input: Any) -> output: Any`"""

ParametricTransform = Callable[[Any, Parameter | ParameterStore], Any]
"""⚙️ any function with the signature `(input: Any, params: Parameter | ParameterStore) -> output: Any`"""

Transform = PureTransform | ParametricTransform
"""➡️ any function with either `PureTransform` or `ParametricTransform` signature"""


def noop(input: Any) -> Any:  # noqa: ANN401, A002
    """🤷 no-operation. Passes through the input.

    Args:
        input (Any): any input

    Returns:
        Any: unmodified input
    """
    return input


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
        transform (Transform): transformation
        TODO: add params arg
        name (str | None, optional): node name. If none, use `transform.__name__`. Defaults to None.
        emoji (str, optional): emoji for string representation. Defaults to "➡️".
    """

    @beartype()
    def __init__(  # noqa: D107
        self,
        transform: Transform,
        params: Parameter | ParameterStore | None = None,
        name: str | None = None,
        emoji: str = "➡️",
    ) -> None:
        super().__init__()
        self.transform = transform
        self.params: ParameterStore | None = ParameterStore([params]) if isinstance(params, Parameter) else params

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
        output = self.transform(input, self.params) if self.params else self.transform(input)

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
    """🪴 branching node with branches as `self.children`.

    When traversed, a random branch is selected based on `probs`.

    Args:
        branches (list[TransformNode]): list of child nodes representing the different branches
        probs (Parameter | None, optional): parameter containing the probabilities for selecting each branch.
            `probs.value` is passed to `numpy.random.choice` as parameter `p`, which is documented as:
            (1-D array_like, optional) the probabilities associated with each entry in a.
            If not given the sample assumes a uniform distribution over all entries. Defaults to None.
        name (str, optional): name of the node. Important for fetching the `probs` if they are not present. Defaults to "branching point".
        seed (int, optional): random number generator seed. Defaults to 42.
    """

    def __init__(
        self, branches: list[TransformNode], probs: Parameter | None = None, name: str = "branching point", seed: int = 42
    ) -> None:  # noqa: D107
        self.name = name
        self.children = [node if isinstance(node, TransformNode) else TransformNode(node) for node in branches]
        self.probs = probs
        self.rng = np.random.default_rng(seed)

        super().__init__(transform=noop, name=name, emoji="🪴")

    def traverse(self, input: Any) -> tuple[Any, anytree.NodeMixin]:  # noqa: A002, ANN401
        """🏃🎲 `streamgen.transforms.Traverse` protocol `(input: Any) -> (output, anytree.NodeMixin | None)`.

        During traversal, a branching node samples the next node from its children.

        Args:
            input (Any): any input

        Returns:
            tuple[Any, anytree.NodeMixin | None]: output and next node to traverse
        """
        idx = self.rng.choice(len(self.children), p=self.probs.value if self.probs else None)
        next_node = self.children[idx]

        return input, next_node

    def update(self) -> None:
        """🆙 updates every parameter."""
        if self.probs:
            self.probs.update()

        for node in self.children:
            node.update()

    def fetch_params(self, params: ParameterStore) -> None:
        """⚙️ fetches params from a ParameterStore.

        Args:
            params (ParameterStore): _description_
        """
        if self.probs is None and self.name in params.scopes:
            probs = list(params.get_scope(self.name).parameters.values())
            assert (
                len(probs) == 1
            ), f'Make sure to only have a single parameter in the scope "{self.name}" when setting the parameters of a `BranchingNode` through `fetch_params`.'  # noqa: E501, S101
            self.probs = probs[0]

        for node in self.children:
            node.fetch_params(params)
