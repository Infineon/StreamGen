"""🎲 this module contain implementations for different samplers.

Samplers are objects that represent distributions.
"""

from typing import Any, Protocol


class Sampler(Protocol):
    """📊 sampler protocol `() -> sample`."""

    def sample() -> Any:  # noqa: D102, ANN401
        ...
