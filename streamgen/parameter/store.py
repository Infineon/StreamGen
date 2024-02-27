"""🗃️ parameter store are dictionary-like collections of parameters and scheudles."""

from typing import Any

import pandas as pd
from beartype import beartype

from streamgen.parameter import Parameter, ParameterDict


class ParameterStore:
    """🗃️ a dictionary-like container of `streamgen.parameter.Parameter` with their names as keys.

    Args:
        parameters (list[Parameter] | ParameterDict): parameters to store

    Examples:
        >>> store = ParameterStore(
                {
                "var1": {
                        "value": 1,
                        "schedule": [2,3],
                        "strategy": "cycle",
                    },
                "var2": {
                        "name": "var2", # can be present, but is not needed
                        "schedule": [0.1, 0.2, 0.3],
                    },
                }
            )

        >>> store = ParameterStore([
                Parameter("var1", 1, [2]),
                Parameter("var2", schedule=[0.0, 1.0]),
            ]
        })
    """

    @beartype()
    def __init__(self, parameters: list[Parameter] | ParameterDict) -> None:  # noqa: D107
        match parameters:
            case list(parameters):
                self.parameters: dict[str, Parameter] = {p.name: p for p in parameters}
            case dict(parameters):
                self.parameters: dict[str, Parameter] = {name: Parameter(**kwargs) for name, kwargs in parameters.items()}

    @beartype()
    def __getitem__(self, name: str) -> Any:  # noqa: ANN401
        """Gets a parameter by its name using `store[name]` syntax.

        Args:
            name (str): name of the parameter

        Returns:
            Any: the `value` of the parameter
        """
        return self.parameters[name].value

    def get_params(self) -> dict[str, Any]:
        """📖 get the current parameters as one dictionary.

        Returns:
            dict[str, Any]: dictionary with `parameter.name` as keys and `parameter.value` as values
        """
        return {name: param.value for name, param in self.parameters.items()}

    def update(self) -> dict[str, Any]:
        """🆙 updates every parameter and returns them using `ParameterStore.get_params`.

        Returns:
            dict[str, Any]: dictionary with `parameter.name` as keys and `parameter.value` as values
        """
        for param in self.parameters.values():
            param.update()

        return self.get_params()

    @beartype()
    def to_dataframe(self, number_of_update_steps: int = 0) -> pd.DataFrame:
        """📅🐼 rolls out `number_of_update_steps` and returns the result as a `pd.DataFrame`.

        Args:
            number_of_update_steps (int, optional): number of rows in the dataframe. Defaults to 1.

        Returns:
            pd.DataFrame: parameter store as dataframe
        """
        return pd.DataFrame(
            [self.get_params()] + ([self.update() for _ in range(number_of_update_steps)] if number_of_update_steps > 0 else []),
        )

    def __str__(self) -> str:
        """🏷️ Returns the string representation `str(self)`.

        Returns:
            str: string representation of self
        """
        s = "🗃️ = {\n"
        for param in self.parameters.values():
            s += "\t" + str(param) + "\n"
        s += "}"
        return s
