"""🌳 sampling trees are trees of transformations that you can traverse from root to leaf to create samples."""

import itertools
from collections.abc import Callable
from copy import deepcopy
from itertools import pairwise
from pathlib import Path
from typing import Any, Self

import anytree
import numpy as np
from anytree.exporter import UniqueDotExporter
from beartype import beartype
from graphviz import Source
from IPython.display import SVG
from IPython.utils import io
from matplotlib import animation
from matplotlib import pyplot as plt
from pandas import DataFrame
from rich.progress import track

from streamgen.enums import SamplingStrategy, SamplingStrategyLit
from streamgen.nodes import ClassLabelNode, SampleBufferNode, TransformNode
from streamgen.parameter import Parameter
from streamgen.parameter.store import ParameterStore
from streamgen.samplers import Sampler
from streamgen.transforms import noop


class BranchingNode(TransformNode):
    """🪴 node with multiple children/branches.

    When traversed, a random branch is selected based on the probabilities defined by `probs`.

    Args:
        branches (dict): dictionary, where each key:value pair represent label:branch.
        probs (Parameter | list[float] None, optional): parameter containing the probabilities for selecting each branch.
            `probs.value` is passed to `numpy.random.choice` as parameter `p`, which is documented as:
            (1-D array_like, optional) the probabilities associated with each entry in a.
            If not given the sample assumes a uniform distribution over all entries. Defaults to None.
        name (str | None, optional): name of the node. Important for fetching the `probs` if not present. Defaults to "branching_node".
        seed (int, optional): random number generator seed. Defaults to 42.
        string_node (Callable[[str], TransformNode], optional): `TransfromNode` constructor from strings used in `construct_tree`.
            Defaults to `ClassLabelNode`.
    """

    def __init__(  # noqa: D107
        self,
        branches: dict,
        probs: Parameter | list[float] | None = None,
        name: str | None = None,
        seed: int = 42,
        string_node: Callable[[str], TransformNode] = ClassLabelNode,
    ) -> None:
        self.name = name if name else "branching_node"

        if isinstance(probs, list):
            probs = Parameter(name="probs", value=probs)

        self.probs = probs
        self.rng = np.random.default_rng(seed)

        self.branches = {branch_name: construct_tree(nodes, string_node) for branch_name, nodes in branches.items()}

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
def construct_tree(
    nodes: Callable | TransformNode | dict | str | list[Callable | TransformNode | dict | str],
    string_node: Callable[[str], TransformNode] = ClassLabelNode,
) -> list[TransformNode]:
    """🏗️ assembles and links nodes into a tree.

    The following rules apply during construction:

    1. Nodes are linked sequentially according to the ordering in the top-level list.
    2. `TransformNode` and sub-classes are not modified.
    3. `Callable`s are cast into `TransformNode`s.
    4. `str` are passed to the `string_node` constructor, which allows to configure which Node type is used for them.
    5. dictionaries are interpreted as `BranchingNode`s, where each value represents a branch.
        The keys `name`, `probs` and `seed` are reserved to describe the node itself.
    6. If there is a node after a `BranchingNode`, then every branch will be connected to a **copy** of this node.
        This ensures that the structure of the tree is preserved (Otherwise we would create a more generic directed acyclic graph),
        which is not supported by `anytree`.

    Args:
        nodes (Callable | TransformNode | dict | str | list[Callable | TransformNode | dict | str]): short-hand description of a tree.
        string_node (Callable[[str], TransformNode], optional): `TransfromNode` constructor from strings used in `construct_tree`.
            Defaults to `ClassLabelNode`.

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
                graph.append(BranchingNode(node, name=name, probs=probs, seed=seed, string_node=string_node))
            case str():
                graph.append(string_node(node))

    # connect the nodes to enable traversal and parameter fetching and updating
    for node, next_node in pairwise(graph):
        match (node, next_node):
            case (BranchingNode(), _):
                # * This is a special shorthand conveninence behaviour:
                # * when we sequentially combine a `BranchingNode` with another node,
                # * we add the other node to every leaf of the branches in the `BranchingNode`
                for branch in node.branches.values():
                    # * the copy operation is needed, since `anytree` does not allow merged branches
                    # * (merged branches are different branches with a common child -> creates a DAG instead of a tree).
                    # * we have to add the copy to the leaf's children to enable traversal
                    for leaf in branch[-1].leaves:
                        next_node_copy = deepcopy(next_node)
                        leaf.children = [next_node_copy]
                        # * we have to add the copy to the branch to handle parameter fetching and updating
                        branch.append(next_node_copy)
            case (_, _):
                node.children = [next_node]

    return graph


class SamplingTree(Sampler):
    """🌳 a tree of `TransformNode`s, that can be sampled from.

    The tree will be constructed using `streamgen.nodes.construct_tree(nodes, string_node)`.

    Args:
        nodes (list[Callable | TransformNode | dict| str]): pythonic short-hand description of a graph/tree.
            `streamgen.samplers.tree.construct_tree` will be called to construct the tree.
        params (ParameterStore | DataFrame | None, optional): parameter store containing additional parameters
            that are passed to the nodes based on the scope. Dataframes will be converted to `ParameterStore`. Defaults to None.
        collate_func (Callable[[list[Any]], Any] | None, optional): function to collate samples when using `self.collect(num_samples)`.
            If None, return a list of samples. Defaults to None.
        string_node (Callable[[str], TransformNode], optional): `TransfromNode` constructor from strings used in `construct_tree`.
            Defaults to `ClassLabelNode`.
    """

    def __init__(  # noqa: D107
        self,
        nodes: list[Callable | TransformNode | dict | str],
        params: ParameterStore | DataFrame | None = None,
        collate_func: Callable[[list[Any]], Any] | None = None,
        string_node: Callable[[str], TransformNode] = ClassLabelNode,
    ) -> None:
        self.nodes = construct_tree(nodes, string_node)

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

    def collect(self, num_samples: int, strategy: SamplingStrategy | SamplingStrategyLit = "stochastic") -> Any:  # noqa: ANN401
        """🪺 collect and concatenate `num_samples` using `sample() and `self.collate_func`.

        Args:
            num_samples (int): number of samples to collect.
                When using the "stochastic" (default) strategy, this refers to the total number of samples.
                When using the "balanced" strategies, this refers to the number of samples per path through the tree.
            strategy (SamplingStrategy | SamplingStrategyLit, optional): sampling strategy. Defaults to "stochastic".

        Returns:
            Any: collection of samples.
                If `self.collate_func` is defined, it will be mapped to the tuple elements in each sample.
                Otherwise this functions just returns a list of samples.
        """
        match strategy:
            case SamplingStrategy.STOCHASTIC:
                samples = [self.sample() for _ in track(range(num_samples), description="🎲 sampling...")]
            case SamplingStrategy.BALANCED:
                paths = self.get_paths()
                samples = list(itertools.chain(*[path.collect(num_samples) for path in paths]))
            case SamplingStrategy.BALANCED_PRUNED:
                paths = self.get_paths(prune=True)
                samples = list(itertools.chain(*[path.collect(num_samples) for path in paths]))

        return tuple(map(self.collate_func, zip(*samples, strict=True))) if self.collate_func else samples

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

    def to_dotfile(
        self,
        file_path: Path = Path("./tree.dot"),
        plotting_func: Callable[[Any, plt.Axes], plt.Axes] | None = None,
        fps: int = 2,
    ) -> None:
        """🕸️ exports the tree as a `dot` file using [graphviz](https://www.graphviz.org/).

        Args:
            file_path (Path, optional): path of the resulting file. Defaults to "./tree.dot".
            plotting_func (Callable[[Any, plt.Axes], plt.Axes]): function to visualize a single sample.
                The function should take a sample and a `plt.Axes` as arguments.
                It is used to create sample animations for `SampleBufferNode`s.
            fps (int, optional): frames per second for the sample animations. Defaults to 2.
        """
        output_path = file_path.parent

        def _nodeattrfunc(node) -> str:  # noqa: ANN001
            """Builds the node attribute list for graphviz."""
            a = f'label="{node.emoji} {node.name}"'
            match node:
                case BranchingNode():
                    probs = [round(1.0 / len(node.children), 3)] * len(node.children) if node.probs is None else str(node.probs)
                    return a + f' shape=diamond tooltip="{probs}"'
                case ClassLabelNode():
                    return a + " shape=cds"
                case SampleBufferNode():
                    # create animation
                    if plotting_func is None:
                        return a + " shape=box"
                    anim = node.plot(plotting_func, display=False)
                    if anim is None:
                        return a + " shape=box"
                    # save gif
                    gif_path = output_path / f"{node.name}.gif"
                    anim.save(gif_path, writer=animation.PillowWriter(fps=fps))
                    # add gif as background
                    return f'label="" shape=box image="{gif_path.name}" imagescale=true'
                case _:
                    return a + f' tooltip="{node.get_params()!s}"'

        dot = UniqueDotExporter(
            self.root,
            graph="digraph",
            nodeattrfunc=_nodeattrfunc,
        )

        dot.to_dotfile(file_path)

    def to_svg(
        self,
        file_path: Path = Path("./tree"),
        plotting_func: Callable[[Any, plt.Axes], plt.Axes] | None = None,
        fps: int = 2,
    ) -> SVG:
        """📹 visualizes the tree as an svg using [graphviz](https://www.graphviz.org/).

        Args:
            file_path (Path, optional): path of the resulting file. Defaults to "./tree.dot".
            plotting_func (Callable[[Any, plt.Axes], plt.Axes]): function to visualize a single sample.
                The function should take a sample and a `plt.Axes` as arguments.
                It is used to create sample animations for `SampleBufferNode`s.
            fps (int, optional): frames per second for the sample animations. Defaults to 2.

        Returns:
            IPython.display.SVG: svg display of dot visualization
        """
        output_path = file_path.parent
        file_stem = file_path.stem
        dot_path = output_path / (file_stem + ".dot")

        with io.capture_output() as _captured:
            self.to_dotfile(dot_path, plotting_func, fps)
            Source.from_file(dot_path).render(dot_path, format="svg")
        return SVG(filename=(output_path / (file_stem + ".dot.svg")))

    def get_paths(self, prune: bool = False) -> list[Self]:  # noqa: FBT001, FBT002
        """🍃 constructs a deterministic path for each leaf in the tree.

        Args:
            prune (bool, optional): If true, only return paths with probabilities greater than zero. Defaults to false.

        Returns:
            list[Self]: list of `SamplingTree`s without branches.
        """
        paths = []
        for leaf in self.root.leaves:
            pruned_path = False
            if prune:
                # check if all probs leading to the leaf are greater than 0
                for node in leaf.ancestors:
                    if isinstance(node, BranchingNode):
                        # get probability for the path leading to the leaf
                        # to do this, we need to find the index of the branch
                        idx = next(idx for idx, branch in enumerate(node.children) if leaf in branch.leaves)
                        prob = node.probs.value[idx]
                        if prob == 0.0:
                            pruned_path = True
            if not pruned_path:
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
