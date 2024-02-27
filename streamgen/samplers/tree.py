"""🌳 sampling trees are trees of transformations that you can traverse from root to leaf to create samples."""

from typing import Any

from streamgen.transforms import TransformNode


class SamplingTree:
    """🌳 a tree of `TransformNode`s, that can be sampled from.

    Args:
        nodes (list[TransformNode]): list of transformation nodes
    """
    def __init__(self, nodes: list[TransformNode]) -> None:  # noqa: D107
        self.root = nodes[0]
        self.nodes = nodes

        for idx, node in enumerate(self.nodes[:-1]):
            node.children = [self.nodes[idx + 1]]

    def sample(self) -> Any:  # noqa: ANN401
        """🎲 generates a sample by traversing the tree from root to one leaf.

        Returns:
            Any: sample
        """
        node = self.root
        out =  None

        while node is not None:
            out, node = node.traverse(out)

        return out

    def update(self) -> None:
        """🆙 updates every parameter."""
        for node in self.nodes:
            node.update()

    def get_params(self):
        pass
