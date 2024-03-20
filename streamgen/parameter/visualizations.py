"""🖼️ parameter visualization functions."""

from typing import Any

import pandas as pd
from beartype import beartype
from beartype.typing import Self
from loguru import logger
from rich.pretty import pretty_repr

from streamgen.parameter import Parameter, ScopedParameterDict