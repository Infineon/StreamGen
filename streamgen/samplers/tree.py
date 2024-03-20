"""🌳 sampling trees are trees of transformations that you can traverse from root to leaf to create samples."""

from collections.abc import Callable
from copy import deepcopy
from itertools import pairwise
from pathlib import Path
from typing import Any, Self

import anytree
import numpy as np
from anytree.exporter import UniqueDotExporter
from beartype import beartype
from pandas import DataFrame
from rich.progress import track

from streamgen.nodes import ClassLabelNode, TransformNode
from streamgen.parameter import Parameter
from streamgen.parameter.store import ParameterStore
from streamgen.samplers import Sampler
from streamgen.transforms import noop


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

        self.branches = {branch_name: construct_tree(nodes) for branch_name, nodes in branches.items()}

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

    def set_update_step(self, idx: int) -> None:
        """🕐 updates every parameter to a certain update step.

        Args:
            idx (int): parameter update step

        Returns:
            None: this function mutates `self`
        """
        if self.probs:
            self.probs[idx]

        for branch in self.branches.values():
            for node in branch:
                node.set_update_step(idx)

    def fetch_params(self, params: ParameterStore) -> None:
        """⚙️ fetches params from a ParameterStore.

        Args:
            params (ParameterStore): parameter store to fetch the params from
        """
        if self.probs is None and self.name in params.scopes:
            probs = list(params.get_scope(self.name).parameters.values())
            assert (  # noqa: S101
                len(probs) == 1
            ), f'Make sure to only have a single parameter in the scope "{self.name}" when setting the parameters of a `BranchingNode` through `fetch_params`.'  # noqa: E501
            self.probs = probs[0]

        for branch in self.branches.values():
            for node in branch:
                node.fetch_params(params)

    def get_params(self) -> ParameterStore | None:
        """⚙️ collects parameters from every node.

        The parameters are scoped based on the node names.

        Returns:
            ParameterStore | None: parameters from every node. None is there are no parameters.
        """
        store = ParameterStore([])

        if self.probs:
            store.scopes.add(self.name)
            store.parameters[self.name] = {}
            store.parameters[self.name][self.probs.name] = self.probs
            store.parameter_names.add(f"{self.name}.{self.probs.name}")

        for branch in self.branches.values():
            for node in branch:
                if params := node.get_params():
                    store |= params

        return store if len(store.parameter_names) > 0 else None


@beartype()
def construct_tree(nodes: Callable | TransformNode | dict | str | list[Callable | TransformNode | dict | str]) -> list[TransformNode]:
    """🏗️ assembles and links nodes into a tree.

    The following rules apply during construction:

    1. Nodes are linked sequentially according to the ordering in the top-level list.
    2. `TransformNode` and sub-classes are not modified.
    3. `Callable`s are cast into `TransformNode`s.
    4. `str` are cast into `ClassLabelNode`
    5. dictionaries are interpreted as `BranchingNode`s, where each value represents a branch.
        The keys `name`, `probs` and `seed` are reserved to describe the node itself.
    6. If there is a node after a `BranchingNode`, then every branch will be connected to a **copy** of this node.
        This ensures that the structure of the tree is preserved (Otherwise we would create a more generic directed acyclic graph),
        which is not supported by `anytree`.

    Args:
        nodes (Callable | TransformNode | dict | str | list[Callable | TransformNode | dict | str]): pythonic short-hand description of a tree.

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
            case str():
                graph.append(ClassLabelNode(node))

    # connect the nodes to enable traversal
    for node, next_node in pairwise(graph):
        match (node, next_node):
            case (BranchingNode(), _):
                for leaf in node.leaves:
                    # * the copy operation is needed, since `anytree` does not allow merged branches
                    # * (merged branches are different branches with a common child -> creates a DAG instead of a tree).
                    #! the `copy` operation tricks `anytree` into not recognizing the merging.
                    # TODO: I need to check a few things to make sure this is ok:
                    #   1. how is a copy of a node linked to its original node (especially regarding mutation)?
                    #   2. does this approach work recursively with multiple nestings?
                    #       -> no, we need to check recursively for branching points.
                    #          fortunately, anytree makes this very easy with `node.leaves`
                    #   3. how does this affect the printing/representation of the tree?
                    leaf.children = [deepcopy(next_node)]
            case (_, _):
                node.children = [next_node]

    return graph


class SamplingTree(Sampler):
    """🌳 a tree of `TransformNode`s, that can be sampled from.

    The tree will be constructed using `streamgen.nodes.construct_tree(nodes)`.

    Args:
        nodes (list[Callable  |  TransformNode  |  dict]): pythonic short-hand description of a graph/tree
        params (ParameterStore | DataFrame | None, optional): parameter store containing additional parameters
            that are passed to the nodes based on the scope. Dataframes will be converted to `ParameterStore`. Defaults to None.
        collate_func (Callable[[list[Any]], Any] | None, optional): function to collate samples when using `SamplingTree.collect(num_samples)`.
            If None, return a list of samples. Defaults to None.
    """

    def __init__(  # noqa: D107
        self,
        nodes: list[Callable | TransformNode | dict],
        params: ParameterStore | DataFrame | None = None,
        collate_func: Callable[[list[Any]], Any] | None = None,
    ) -> None:
        self.nodes = construct_tree(nodes)

        self.root = self.nodes[0]

        match params:
            case None:
                self.params = ParameterStore([])
            case DataFrame():
                self.params = ParameterStore.from_dataframe(params)
            case ParameterStore():
                self.params = params

        # pass parameters to nodes
        for node in self.nodes:
            node.fetch_params(self.params)

        self.collate_func = collate_func

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

    def __next__(self) -> Any:  # noqa: ANN401
        """🪺 returns the next element during iteration.

        The iterator never runs out of samples, so no `StopIteration` exception is raised.

        Returns:
            Any: a sample
        """
        return self.sample()

    def __iter__(self) -> Self:
        """🏭 turns self into an iterator.

        Required to loop over a `SamplingTree`.
        """
        return self

    def collect(self, num_samples: int) -> Any:  # noqa: ANN401
        """🪺 collect and concatenate `num_samples` using `sample() and `self.collate_func`.

        Args:
            num_samples (int): number of samples to collect

        Returns:
            Any: collection of samples
        """
        samples = [self.sample() for _ in track(range(num_samples), description="🎲 sampling...")]

        return self.collate_func(samples) if self.collate_func else samples

    def update(self) -> None:
        """🆙 updates every parameter."""
        for node in self.nodes:
            node.update()

    def set_update_step(self, idx: int) -> None:
        """🕐 updates every parameter to a certain update step using `param[idx]`.

        Args:
            idx (int): parameter update step

        Returns:
            None: this function mutates `self`
        """
        for node in self.nodes:
            node.set_update_step(idx)

    def get_params(self) -> ParameterStore | None:
        """⚙️ collects parameters from every node.

        The parameters are scoped based on the node names.

        Returns:
            ParameterStore | None: parameters from every node. None is there are no parameters.
        """
        store = ParameterStore([])

        for node in self.nodes:
            if params := node.get_params():
                store |= params

        return store if len(store.parameter_names) > 0 else None

    def to_dotfile(self, file_path: Path = Path("./tree.dot")) -> None:
        """🕸️ exports the tree as a `dot` file using [graphviz](https://www.graphviz.org/).

        Args:
            file_path (Path, optional): path of the resulting file. Defaults to "./tree.dot".
        """

        def _nodeattrfunc(node) -> str:  # noqa: ANN001
            """Builds the node attribute list for graphviz."""
            a = f'label="{node!s}"'
            match node:
                case BranchingNode():
                    return a + ",shape=diamond"
                case ClassLabelNode():
                    return a + ",shape=cds"
                case _:
                    return a

        dot = UniqueDotExporter(
            self.root,
            graph="digraph",
            nodeattrfunc=_nodeattrfunc,
        )

        dot.to_dotfile(file_path)

    def get_paths(self) -> list[Self]:
        """🍃 constructs a deterministic path for each leaf in the tree.

        Returns:
            list[Self]: list of `SamplingTree`s without branches.
        """
        paths = []
        for leaf in self.root.leaves:
            path = [node for node in deepcopy(leaf.ancestors) if not isinstance(node, BranchingNode)] + [deepcopy(leaf)]
            for node, next_node in pairwise(path):
                node.children = [next_node]
            paths.append(SamplingTree(path, self.params))

        return paths

    def __str__(self) -> str:
        """🏷️ Returns the string representation `str(self)`.

        Returns:
            str: string representation of self
        """
        s = "🌳\n"
        for pre, _, node in anytree.RenderTree(self.root, style=anytree.ContRoundStyle()):
            s += pre + str(node) + "\n"
        return s
