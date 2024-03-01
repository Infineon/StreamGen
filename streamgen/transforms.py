"""➡️ useful transformations."""

from collections.abc import Callable
from functools import wraps
from typing import Any


def noop(input: Any) -> Any:  # noqa: ANN401, A002
    """🤷 no-operation. Passes through the input.

    Args:
        input (Any): any input

    Returns:
        Any: unmodified input
    """
    return input


def set_value_in_dict(dictionary: dict[str, Any], value: Any, key: str) -> dict:  # noqa: ANN401
    """📖 sets `dictionary[key]` to `value` and returns the result.

    Args:
        dictionary (dict): any dictionary
        value (Any): any value
        key (str): key of the value in `dictionary`. Defaults to "target".

    Returns:
        dict: labeled `dictionary`
    """
    dictionary[key] = value
    return dictionary


def operate_on_key(key: str = "input") -> Callable:
    """🗝️ function decorator, that converts `func(input: dict, ...)` to `func(input[key], ...)`.

    The result is put back into `input`, the first argument to `func`.

    This facilitates working with dictionaries as first arguments.

    Examples:
        >>> @operate_on_key(key="x")
            def add(x, n):
                return x + n

        >>> add({"x": 1, "target": None}, 2)
        {'x': 3, 'target': None}

    Args:
        key (str, optional): key for fetching the value from `input`. Defaults to "input".

    Returns:
        Callable: modified function
    """

    def decorator(func):  # noqa: ANN202, ANN001
        @wraps(func)
        def wrapper(input, *args, **kwargs):  # noqa: ANN202, ANN001, A002, ANN002, ANN003
            x = input[key]
            out = func(x, *args, **kwargs)
            input[key] = out
            return input

        return wrapper

    return decorator
