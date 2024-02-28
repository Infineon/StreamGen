"""🧪 `streamgen.parameter.Parameter` tests."""
# ruff: noqa: S101, D103, ANN001, ANN201, PLR2004

import numpy as np
from loguru import logger

from streamgen.parameter import Parameter
from streamgen.parameter.store import ParameterStore


def test_parameter_empty_list_initialization() -> None:
    """Tests the empty list initialization behavior of a parameter store."""
    store = ParameterStore([])

    assert store.parameters == {}
    assert store.parameter_names == set()
    assert store.scopes == set()


def test_parameter_list_initialization() -> None:
    """Tests the list initialization behavior of a parameter store."""
    params = [
        Parameter("var1", 1, [2]),
        Parameter("var2", schedule=[0.0, 1.0], emoji="📉"),
    ]

    store = ParameterStore(params)

    logger.debug("\n" + str(store))
    assert str(store) == "🗃️ = {⚙️ var1: 1, 📉 var2: 0.0}"

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
    }

    store = ParameterStore(params)

    assert store["var1"].value == 1
    assert store["var2"].value == 0.0

    store.update()

    assert store["var1"].value == 2
    assert store["var2"].value == 1.0


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
        "scope1": {
            "var1": {  # var1 can be used again since its inside a scope
                "value": -1,
                "schedule": [-2, -3],
                "strategy": "cycle",
            },
        },
    }

    store = ParameterStore(params)

    assert store["var1"].value == 1
    assert store["var2"].value == 0.1
    assert store["scope1.var1"].value == -1
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
