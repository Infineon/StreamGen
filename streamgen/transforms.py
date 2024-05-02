"""➡️ useful transformations."""

from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np


class LabelEncoder:
    """🏷️ index of class-name lookup transform.

    Args:
        classes (List[str]): list of labels/class-names. Order determines the indices.

    Example:
        >>> LabelEncoder(["A", "B", "C"])("C")
        array(2, dtype=int64)
    """

    def __init__(self, classes: list[str]) -> None:  # noqa: D107
        self.classes = {class_: idx for idx, class_ in enumerate(classes)}

    def __call__(self, label: str) -> np.ndarray:
        """Standard call method.

        Args:
            label (str): class name

        Returns:
            np.ndarray: index encoded target array
        """
        return np.array(self.classes[label], dtype=np.int64)


class LabelDecoder:
    """🏷️ label/class-name for index lookup transform.

    Args:
        classes (List[str]): list of labels/class-names. Order determines the indices.

    Example:
        >>> LabelDecoder(["A", "B", "C"])(1)
        'B'
    """

    def __init__(self, classes: list[str]) -> None:  # noqa: D107
        self.classes = classes

    def __call__(self, idx: int) -> str:
        """Standard call method.

        Args:
            idx (int): index of the target class

        Returns:
            str: class name
        """
        return self.classes[idx]


class MultiLabelEncoder:
    """🏷️ encoding of multiple labels/class-names.

    Args:
        classes (List[str]): list of labels/class-names. Order determines the indices.

    Example:
        >>> MultiLabelEncoder(["A", "B", "C"])(["A", "C"])
        array([1., 0., 1.], dtype=float32)
    """

    def __init__(self, classes: list[str]) -> None:  # noqa: D107
        self.classes = classes

    def __call__(self, labels: list[str]) -> np.ndarray:
        """Standard call method.

        Args:
            labels (list[str]): (potentially empty) list of class_ names.

        Returns:
            np.ndarray: encoded array of targets
        """
        return np.array([int(p in labels) for p in self.classes], dtype=np.float32)


class MultiLabelDecoder:
    """🏷️ decoding of labels/class-names.

    Args:
        classes (List[str]): list of labels/class-names. Order determines the indices.

    Example:
        >>> MultiLabelDecoder(["A", "B", "C"])([1.0, 0.0, 1.0])
        array([1., 0., 1.], dtype=float32)
    """

    def __init__(self, classes: list[str]) -> None:  # noqa: D107
        self.classes = classes

    def __call__(self, labels: list[int | float]) -> list[str]:
        """Standard call method.

        Args:
            labels (list[int | float]): encoded array of targets.
                must have the same length as `self.classes`.

        Returns:
            list[str]: (potentially empty) list of class_ names
        """
        return [p for i, p in enumerate(self.classes) if bool(labels[i])]


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


def operate_on_index(idx: int = 0) -> Callable:
    """#️⃣ function decorator, that converts `func(input: Sequence, ...)` to `func(input[idx], ...)`.

    The other parts of the input sequence are not passed to the function, but still returned by the decorator.

    This facilitates working with sequences (like tuples) as first arguments.

    Examples:
        >>> @operate_on_index()
            def add(x, n):
                return x + n

        >>> add((1, "target"), 2)
        (3, "target")

    Args:
        idx (int, optional): index for fetching the value from `input`. Defaults to 0 (the first element).

    Returns:
        Callable: modified function
    """

    def decorator(func):  # noqa: ANN202, ANN001
        @wraps(func)
        def wrapper(input, *args, **kwargs):  # noqa: ANN202, ANN001, A002, ANN002, ANN003
            input_type = type(input)
            input = list(input)  # noqa: A001
            x = input[idx]
            out = func(x, *args, **kwargs)
            input[idx] = out
            return input_type(input)

        return wrapper

    return decorator
