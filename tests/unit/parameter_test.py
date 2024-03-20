"""🧪 `streamgen.parameter.Parameter` tests."""
# ruff: noqa: S101, D103, ANN001, ANN201, PLR2004

from itertools import count

import numpy as np
import pytest

from streamgen.exceptions import ParameterOutOfRangeError
from streamgen.parameter import Parameter


def test_constant_parameter() -> None:
    """Tests the behavior of a parameter without a schedule."""
    param = Parameter("test", 42, None)

    assert param.name == "test"
    assert str(param) == "test=42"
    assert param.value == 42, "initial value should be 42"
    assert param.update() == 42, "1st updated value should be 42 due to the `hold` strategy"


def test_list_parameter() -> None:
    """Tests the behavior of a parameter with a list schedule."""
    param = Parameter("test", 1, [2])

    assert param.value == 1, "initial value should be 1"
    assert param.update() == 2, "1st updated value should be 2 as defined by schedule"
    assert param.update() == 2, "2nd updated value should be 2 due to the `hold` strategy"


def test_cyclic_parameter() -> None:
    """Tests the behavior of a parameter with a cyclic list schedule."""
    param = Parameter("test", schedule=[1, 2], strategy="cycle")

    assert param.value == 1, "initial value should be 1"
    assert param.update() == 2, "1st updated value should be 2 as defined by schedule"
    assert param.update() == 1, "2nd updated value should be 1 due to the `cycle` strategy"


def test_parameter_out_of_range() -> None:
    """Tests the behavior of a parameter with the `raise exception` strategy."""
    param = Parameter("test", 1, [2], strategy="raise exception")

    assert param.update() == 2, "1st updated value should be 2 as defined by schedule"

    with pytest.raises(ParameterOutOfRangeError):
        param.update()


def test_numpy_parameter() -> None:
    """Tests the use of `numpy.array`s as parameters and schedules."""
    param = Parameter("test", schedule=np.linspace(0.0, 1.0, 3))

    assert param.value == 0.0, "initial value should be 0.0"
    assert param.update() == 0.5, "1st updated value should be 0.5 as defined by schedule"
    assert param.update() == 1.0, "2nd updated value should be 1.0 as defined by schedule"

    param = Parameter("array", schedule=np.arange(6).reshape(2, 3))

    assert param.value.shape == (3,), "iterating over a `numpy.array` should only iterate over the first dimension"


def test_generator_parameter() -> None:
    """Tests the use of parameters with generator functions as schedules."""
    param = Parameter("count", schedule=count())

    assert param.value == 0, "initial value should be 0"
    assert param.update() == 1, "1st updated value should be 1 as defined by schedule"


def test_parameter_indexing() -> None:
    """Tests the explicit updating of a parameter via indexing."""
    param = Parameter("count", schedule=count())

    assert param.value == 0, "initial value should be 0"
    assert param.update() == 1, "1st updated value should be 1 as defined by schedule"

    assert param[0] == 0, "initial value should be 0, even after updating"
    assert param.value == 0, "initial value should be 0, even after updating"
    assert param.update() == 1, "1st updated value should be 1 as defined by schedule"
    assert param[2] == 2, "using indexing should reset the schedule"
    assert param.value == 2, "using indexing should reset the schedule"
