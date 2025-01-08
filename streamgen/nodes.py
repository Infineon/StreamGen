"""🪢 different node implementations using [anytree](https://anytree.readthedocs.io/en/stable/) `NodeMixin`."""

import inspect
from collections import deque
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import anytree
import IPython
import matplotlib.pyplot as plt
import seaborn as sns
from beartype import beartype
from loguru import logger
from matplotlib import animation

from streamgen.enums import ArgumentPassingStrategy, ArgumentPassingStrategyLit
from streamgen.parameter import Parameter
from streamgen.parameter.store import ParameterStore
from streamgen.transforms import noop


@runtime_checkable
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
    def __init__(  # noqa: D107
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

    def set_update_step(self, idx: int) -> None:
        """🕐 updates every parameter of `self.params` to a certain update step using `self.params.set_update_step`.

        Args:
            idx (int): parameter update step

        Returns:
            None: this function mutates `self`
        """
        if self.params:
            self.params.set_update_step(idx)

    def fetch_params(self, params: ParameterStore) -> None:
        """⚙️ fetches params from a ParameterStore.

        The parameters are fetched from both a matching scope and
        the top-level/global scope with the scope having precedence.

        Skips fetching if the node was explicitly parameterized.

        Args:
            params (ParameterStore): _description_
        """
        if self.params:
            return
        self.params = ParameterStore()
        if self.name in params.scopes:
            self.params = params.get_scope(self.name)

        # infer missing arguments that were not present in the scope of the transform
        missing_arguments = [
            param.name for param in inspect.signature(self.transform).parameters.values() if param.name not in self.params.parameter_names
        ]
        # if those missing arguments are in the top-level scope, add those parameters
        for param_name in missing_arguments:
            if param_name in params.parameter_names:
                self.params[param_name] = params[param_name]

    def get_params(self) -> ParameterStore | None:
        """⚙️ returns current parameters.

        Returns:
            ParameterStore | None: parameters. None is there are no parameters.
        """
        if self.params is None:
            return None

        store = ParameterStore([])
        store.scopes.add(self.name)
        store.parameters[self.name] = {}
        for name, param in self.params.parameters.items():
            store.parameters[self.name][name] = param
            store.parameter_names.add(f"{self.name}.{name}")

        return store

    def __repr__(self) -> str:
        """🏷️ Returns the string representation `str(self)`.

        Returns:
            str: string representation of self
        """
        return f"{self.emoji} `{self.name}({repr(self.params)[1:-1] if self.params else ''})`"


class ClassLabelNode(TransformNode):
    """🏷️ node which sets the class label.

    Args:
        label (str | int): class label or index
        label_func (Callable[[Any, str | int], Any], optional): function which adds the label information.
            Defaults to `lambda input, label: (input, label)`.
    """

    def __init__(  # noqa: D107
        self,
        label: str | int,
        label_func: Callable[[Any, str | int], Any] = lambda input, label: (input, label),
    ) -> None:
        self.label = label
        self.label_func = label_func

        super().__init__(transform=noop, name=f"label={self.label}", emoji="🏷️")

    def traverse(self, input: Any) -> tuple[Any, anytree.NodeMixin]:  # noqa: A002, ANN401
        """🏃🎲 `streamgen.transforms.Traverse` protocol `(input: Any) -> (output, anytree.NodeMixin | None)`.

        During traversal, a label node sets the label in `input` with `self.label_func`.

        Args:
            input (Any): any input

        Returns:
            tuple[Any, anytree.NodeMixin | None]: output and next node to traverse
        """
        output = self.label_func(input, self.label)

        return super().traverse(output)

    def __repr__(self) -> str:
        """🏷️ Returns the string representation `str(self)`.

        Returns:
            str: string representation of self
        """
        return f"🏷️ `{self.label}`"


class SampleBufferNode(TransformNode):
    """🗃️ node which remembers the last samples.

    Args:
        name (str | None, optional): name of the buffer. Defaults to "sample buffer".
        num_samples (int, optional): maximum number of samples to store. Defaults to 4.
    """

    def __init__(  # noqa: D107
        self,
        name: str | None = None,
        num_samples: int = 4,
    ) -> None:
        name = name if name is not None else "sample buffer"
        self.samples = deque(maxlen=num_samples)

        super().__init__(transform=noop, name=name, emoji="🗃️")

    def traverse(self, input: Any) -> tuple[Any, anytree.NodeMixin]:  # noqa: A002, ANN401
        """🏃🎲 `streamgen.transforms.Traverse` protocol `(input: Any) -> (output, anytree.NodeMixin | None)`.

        During traversal, a sample buffer node adds samples to `self.samples`.

        Args:
            input (Any): any input

        Returns:
            tuple[Any, anytree.NodeMixin | None]: output and next node to traverse
        """
        self.samples.append(input)

        return super().traverse(input)

    def _plotting_func_wrapper(self, idx: int, ax: plt.Axes, plotting_func: Callable[[Any, plt.Axes], plt.Axes]) -> None:
        """🖼️ private wrapper for the `plotting_func` argument in `self.plot`.

        Args:
            idx (int): index of the frame. Passed into this function by matplotlib.
            ax (plt.Axes): artist to draw to
            plotting_func (Callable[[Any, plt.Axes], plt.Axes]): function to visualize a single sample.
                The function should take a sample and a `plt.Axes` as arguments
        """
        labeled = isinstance(self.samples[0], tuple)
        if labeled:
            sample, target = self.samples[idx]
        else:
            sample = self.samples[idx]
        ax.clear()
        plotting_func(sample, ax)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    def plot(
        self,
        plotting_func: Callable[[Any, plt.Axes], plt.Axes],
        display: bool = True,  # noqa: FBT001, FBT002
    ) -> IPython.display.HTML | animation.FuncAnimation | None:
        """📹 visualizes the samples in the buffer as an animation.

        Args:
            plotting_func (Callable[[Any, plt.Axes], plt.Axes]): function to visualize a single sample.
                The function should take a sample and a `plt.Axes` as arguments.
            display (bool, optional): If true, wraps the animation object in an `IPython.display.HTML`. Defaults to True.

        Returns:
            IPython.display.HTML | animation.FuncAnimation | None: matplotlib animation object
        """
        if len(self.samples) == 0:
            return None

        sns.set_theme()
        fig, ax = plt.subplots(figsize=(3.2, 2.4))

        anim = animation.FuncAnimation(fig, self._plotting_func_wrapper, frames=len(self.samples), fargs=(ax, plotting_func))

        return IPython.display.HTML(anim.to_jshtml()) if display else anim
