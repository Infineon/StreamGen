"""➡️ useful transformations."""

from typing import Any


def noop(input: Any) -> Any:  # noqa: ANN401, A002
    """🤷 no-operation. Passes through the input.

    Args:
        input (Any): any input

    Returns:
        Any: unmodified input
    """
    return input
