"""🔢 all enumerations for `streamgen`.

For every enumeration, a Literal with the same values as the enumeration is created.
In every function that excepts a custom `streamgen` enum as an argument, use the type
<enum_name> | <enum_name>Lit to have strong type checking, string autocompletion and
good documentation generation.
"""

from enum import Enum
from typing import Literal, TypeAlias


class ParameterOutOfRangeStrategy(str, Enum):
    """❓ strategy which defines what happens when there is no valid next value."""

    HOLD = "hold"
    """Holds/Keeps the last valid value."""
    CYCLE = "cycle"
    """Use the first valid value."""
    RAISE_EXCEPTION = "raise exception"
    """Raises a ParameterOutOfRangeError."""


ParameterOutOfRangeStrategyLit: TypeAlias = Literal[
    "hold",
    "cycle",
    "raise exception",
]
