"""🧪 `streamgen.parameter.Parameter` tests."""
# ruff: noqa: S101, D103, ANN001, ANN201, PLR2004


from streamgen.parameter import Parameter
from streamgen.parameter.store import ParameterStore


def test_parameter_list_initialization() -> None:
    """Tests the list initialization behavior of a parameter store."""
    params = [
        Parameter("var1", 1, [2]),
        Parameter("var2", schedule=[0.0, 1.0]),
    ]

    store = ParameterStore(params)

    assert store["var1"] == 1
    assert store["var2"] == 0.0

    store.update()

    assert store["var1"] == 2
    assert store["var2"] == 1.0


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

    assert store["var1"] == 1
    assert store["var2"] == 0.0

    store.update()

    assert store["var1"] == 2
    assert store["var2"] == 1.0
