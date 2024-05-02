"""🎲 this module contain implementations for different samplers.

Samplers are objects that represent distributions.
"""

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Sampler(Iterator, Protocol):
    """📊 sampler protocol `() -> sample`.

    Sampler also implement the iterator protocol.
    """

    def sample(self) -> Any:  # noqa: ANN401
        """🎲 sample from the `Sampler`s distribution.

        Returns:
            Any: a sample
        """
        ...

    def collect(self, num_samples: int) -> Any:  # noqa: ANN401
        """🪺 collect and concatenate `num_samples` using `sample()`.

        Args:
            num_samples (int): number of samples to collect

        Returns:
            Any: collection of samples
        """
        ...

    def update(self) -> None:
        """🆙 updates every parameter."""
        ...
