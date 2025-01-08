"""🧪 `streamgen.parameter.Parameter` tests."""
# ruff: noqa: S101, D103, ANN001, ANN201, PLR2004

import numpy as np
import pandas as pd
from loguru import logger

from streamgen.parameter import Parameter
from streamgen.parameter.store import ParameterStore


def test_parameter_empty_list_initialization() -> None:
    """Tests the empty list initialization behavior of a parameter store."""
    store = ParameterStore([])

    assert store.parameters == {}
    assert store.parameter_names == set()
    assert store.scopes == set()


def test_parameter_empty_initialization() -> None:
    """Tests the empty list initialization behavior of a parameter store."""
    store = ParameterStore()

    assert store.parameters == {}
    assert store.parameter_names == set()
    assert store.scopes == set()


def test_parameter_list_initialization() -> None:
    """Tests the list initialization behavior of a parameter store."""
    params = [
        Parameter("var1", 1, [2]),
        Parameter("var2", schedule=[0.0, 1.0]),
    ]

    store = ParameterStore(params)

    logger.debug("\n" + str(store))
    assert str(store) == "{'var1': 1, 'var2': 0.0}"

    assert store.parameter_names == {"var1", "var2"}

    assert store["var1"].value == 1
    assert store["var2"].value == 0.0

    store.update()

    assert store["var1"].value == 2
    assert store["var2"].value == 1.0


def test_parameter_dict_initialization() -> None:
    """Tests the dict initialization behavior of a parameter store."""
    params = {
        "var1": {
            "value": 1,
            "schedule": [2, 3],
            "strategy": "cycle",
        },
        "var2": {
            "name": "var2",  # can be present, but is not needed
            "schedule": [0.0, 1.0, 0.3],
        },
        "var3": 42, # shorthand for a parameter without a schedule
    }

    store = ParameterStore(params)

    assert store["var1"].value == 1
    assert store["var2"].value == 0.0
    assert store["var3"].value == 42

    store.update()

    assert store["var1"].value == 2
    assert store["var2"].value == 1.0


def test_parameter_dict_setitem() -> None:
    """Tests the dynamic setting of parameters."""
    params = {
        "var1": {
            "value": 1,
            "schedule": [2, 3],
            "strategy": "cycle",
        },
        "var2": {
            "name": "var2",  # can be present, but is not needed
            "schedule": [0.0, 1.0, 0.3],
        },
    }

    store = ParameterStore(params)

    store["var3"] = Parameter("var3", 42)
    store["nested.var4"] = (1,2)

    assert store.scopes == {"nested"}
    assert store.parameter_names == {"var1", "var2", "var3", "nested.var4"}


def test_updating_parameters_by_index() -> None:
    """Tests the explicit updating of every parameter using indexing with integers."""
    params = {
        "var1": {
            "value": 1,
            "schedule": [2, 3],
            "strategy": "cycle",
        },
        "var2": {
            "name": "var2",  # can be present, but is not needed
            "schedule": [0.0, 1.0, 0.3],
        },
    }

    store = ParameterStore(params)

    assert store["var1"].value == 1
    assert store["var2"].value == 0.0

    store.update()

    assert store["var1"].value == 2
    assert store["var2"].value == 1.0

    store.set_update_step(0)

    assert store["var1"].value == 1
    assert store["var2"].value == 0.0

    store.update()

    assert store["var1"].value == 2
    assert store["var2"].value == 1.0

    store.set_update_step(2)

    assert store["var1"].value == 3
    assert store["var2"].value == 0.3


def test_scoped_parameter_dict_initialization() -> None:
    """Tests the dict initialization behavior of a parameter store."""
    params = {
        "var1": {
            "value": 1,
            "schedule": [2, 3],
            "strategy": "cycle",
        },
        "var2": {
            "name": "var2",  # can be present, but is not needed
            "schedule": [0.1, 0.2, 0.3],
        },
        "var3": 42, # shorthand for a parameter without a schedule
        "scope1": {
            "var1": {  # var1 can be used again since its inside a scope
                "value": -1,
                "schedule": [-2, -3],
                "strategy": "cycle",
            },
            "var2": 42, # shorthand for a parameter without a schedule
        },
    }

    store = ParameterStore(params)

    assert store["var1"].value == 1
    assert store["var2"].value == 0.1
    assert store["var3"].value == 42
    assert store["scope1.var1"].value == -1
    assert store["scope1.var2"].value == 42
    assert store.scopes == {"scope1"}

    store.update()

    assert store["var1"].value == 2
    assert store["var2"].value == 0.2
    assert store["scope1.var1"].value == -2


def test_to_dataframe():
    """Tests the rollout as a dataframe."""
    params = {
        "var1": {
            "value": 1,
            "schedule": [2, 3],
            "strategy": "cycle",
        },
        "var2": {
            "name": "var2",  # can be present, but is not needed
            "schedule": [0.0, 0.5, 1.0],
        },
        "array": {
            "schedule": np.arange(6).reshape((3, 2)),
        },
    }

    store = ParameterStore(params)

    # with `number_of_updates=0`, no updates are performed
    df = store.to_dataframe()
    logger.debug(f"df\n{df}")

    assert len(df) == 1
    assert df["var1"][0] == 1
    assert df["var2"][0] == 0.0

    df = store.to_dataframe(number_of_update_steps=2)
    logger.debug(f"df\n{df}")

    assert len(df) == 3
    assert df["var1"][1] == 2
    assert df["var2"][1] == 0.5

    # now test if state has changed.
    # the store should be at step 3
    df = store.to_dataframe()

    assert len(df) == 1
    assert df["var1"][0] == 3
    assert df["var2"][0] == 1.0


def test_from_dataframe():
    """Tests construction from a dataframe."""
    df = pd.DataFrame(
        {
            "var1": [1, 2, 3],
            "var2": [4, 5, 6],
            "scope1.var1": [-1, -2, -3],
            "scope2.var2": [-4, -5, -6],
        },
    )

    store = ParameterStore.from_dataframe(df)

    assert store["var1"].value == 1
    assert store["var2"].value == 4
    assert store["scope1.var1"].value == -1
    assert store.scopes == {"scope1", "scope2"}

    store.update()

    assert store["var1"].value == 2
    assert store["var2"].value == 5
    assert store["scope1.var1"].value == -2


def test_combining_parameters():
    """Tests combining parameters and parameter stores with `|` and `|=`."""
    param1 = Parameter("a", 1)
    param2 = Parameter("b", 2)
    param3 = Parameter("c", 3)
    store2 = ParameterStore(
        {
            "scope1": {
                "d": {"value": 4},
            },
        },
    )

    store = param1 | param2
    assert store.parameter_names == {"a", "b"}
    assert store.scopes == set()

    # |= also works
    param1 |= param2
    assert isinstance(param1, ParameterStore)

    store |= param3

    assert "c" in store.parameter_names

    store |= store2
    assert store.scopes == {"scope1"}
