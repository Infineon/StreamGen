"""⚙️ parameters are variables that change over time according to a schedule."""
from __future__ import annotations

from collections.abc import Iterable
from itertools import cycle
from typing import Generic, TypedDict, TypeVar

from beartype import beartype

from streamgen.enums import (
    ParameterOutOfRangeStrategy,
    ParameterOutOfRangeStrategyLit,
)
from streamgen.exceptions import ParameterOutOfRangeError

T = TypeVar("T")


@beartype()
class Parameter(Generic[T]):
    """⚙️ parameters are variables that change over time according to a schedule.

    Args:
        name (str): variable name of the parameter. Defaults to "param".
        value (Generic[T] | None): the value of the parameter.
            if None and schedule is defined, use the first value of the schedule.
            Defaults to None.
        schedule (list[T] | Schedule[T] | None): a schedule for the parameter.
            Defaults to None.
        parameter_out_of_range_strategy
            (ParameterOutOfRangeStrategy | ParameterOutOfRangeStrategyLit):
            strategy which defines what happens when calling `update` and there is
            no valid next value available. Defaults to `"hold"`, where the last
            valid value is held (not updated).
        emoji (str): emoji used for the str representation of the parameter. Defaults to ⚙️.
    """

    def __init__(  # noqa: D107, PLR0913
        self,
        name: str = "param",
        value: T | None = None,
        schedule: Iterable[T] | None = None,
        strategy: ParameterOutOfRangeStrategy | ParameterOutOfRangeStrategyLit = "hold",
        emoji: str = "⚙️",
    ) -> None:
        assert "." not in name, "`.` in parameter names are reserved for scopes."  # noqa: S101
        self.name = name
        self.emoji = emoji
        self.value = value
        self.schedule = iter(schedule) if schedule is not None else None

        self.strategy = strategy

        if strategy == ParameterOutOfRangeStrategy.CYCLE:
            self.schedule = cycle(self.schedule)

        if self.value is None and self.schedule:
            self.value = self.update()

    def update(self) -> T:
        """🆙 updates the value according to the schedule and strategy.

        Returns:
            T: updated value

        Raises:
            ParameterOutOfRangeError: when an `update` leads to an invalid value.
        """
        if self.schedule is None:
            return self.value

        try:
            self.value = next(self.schedule)
        except StopIteration as err:
            if self.strategy == ParameterOutOfRangeStrategy.RAISE_EXCEPTION:
                raise ParameterOutOfRangeError from err

        return self.value

    def __str__(self) -> str:
        """🏷️ Returns the string representation `str(self)`.

        Returns:
            str: string representation of self
        """
        return f"{self.emoji} {self.name}: {self.value}"


class ParameterDict(Generic[T], TypedDict, total=False):
    """📖 typed dictionary of `streamgen.parameter.Parameter`."""

    name: str | None
    value: T | None
    schedule: Iterable[T] | None
    strategy: ParameterOutOfRangeStrategy | ParameterOutOfRangeStrategyLit | None
    emoji: str


ScopedParameterDict = dict[str, ParameterDict | dict[str, ParameterDict]]
"""🔭📖 representation of multiple `streamgen.parameter.Parameter` as a dictionary.

The dictionary can be nested one level to create parameter scopes.
All top-level parameters are considered as having `scope=None`.

Examples:
    >>> params = {
            "var1": {
                "value": 1,
                "schedule": [2,3],
                "strategy": "cycle",
            },
            "var2": {
                "name": "var2", # can be present, but is not needed
                "schedule": [0.1, 0.2, 0.3],
            },
            "scope1": {
                "var1": { # var1 can be used again since its inside a scope
                    "value": 1,
                    "schedule": [2,3],
                    "strategy": "cycle",
                },
            },
        }
"""
