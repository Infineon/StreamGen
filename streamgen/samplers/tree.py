"""🌳 sampling trees are trees of transformations that you can traverse from root to leaf to create samples."""

from typing import Any

import anytree

from streamgen.parameter.store import ParameterStore
from streamgen.transforms import Transform, TransformNode


class SamplingTree:
    """🌳 a tree of `TransformNode`s, that can be sampled from.

    Args:
        nodes (list[Transform | TransformNode]): list of transforms or transformation nodes
        params (ParameterStore | None, optional): parameter store containing additional parameters
            that are passed to the nodes based on the scope. Defaults to None.
    """

    def __init__(self, nodes: list[Transform | TransformNode], params: ParameterStore | None = None) -> None:  # noqa: D107
        # if callables adhering to the `Transform` signature are passed, transform them to `TransformNode`
        # TODO: add shorthand parsing (lists -> Decision Nodes, first str in nested list is decision node name)
        self.nodes = [node if isinstance(node, TransformNode) else TransformNode(node) for node in nodes]
        self.root = self.nodes[0]
        self.params = params if params else ParameterStore([])

        # pass parameters to nodes
        for node in self.nodes:
            node.fetch_params(self.params)

        # connect the nodes to enable traversal
        for idx, node in enumerate(self.nodes[:-1]):
            node.children = [self.nodes[idx + 1]]

    def sample(self) -> Any:  # noqa: ANN401
        """🎲 generates a sample by traversing the tree from root to one leaf.

        Returns:
            Any: sample
        """
        node = self.root
        out = None

        while node is not None:
            out, node = node.traverse(out)

        return out

    def update(self) -> None:
        """🆙 updates every parameter."""
        for node in self.nodes:
            node.update()

    def get_params(self) -> ParameterStore | None:
        """⚙️ collects parameters from every node.

        The parameters are scoped based on the node names.

        Returns:
            ParameterStore | None: parameters from every node. None is there are no parameters.
        """
        if all(node.params is None for node in self.nodes):
            return None

        store = ParameterStore([])

        for node in self.nodes:
            if node.args:
                scope = node.name
                store.scopes.add(scope)
                store.parameters[scope] = {param.name: param for param in node.params.parameters.values()}
                store.parameter_names.extend([f"{scope}.{param.name}" for param in node.params.parameters.values()])

        return store

    def __str__(self) -> str:
        """🏷️ Returns the string representation `str(self)`.

        Returns:
            str: string representation of self
        """
        s = "🌳\n"
        for pre, _, node in anytree.RenderTree(self.root, style=anytree.ContRoundStyle()):
            s += pre + str(node) + "\n"
        return s
