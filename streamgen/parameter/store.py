"""🗃️ parameter store are dictionary-like collections of parameters and scheudles."""

from typing import Any

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

    def __init__(self, parameters: list[Parameter] | ParameterDict) -> None:  # noqa: D107
        match parameters:
            case list(parameters):
                self.parameters: dict[str, Parameter] = {p.name: p for p in parameters}
            case dict(parameters):
                self.parameters: dict[str, Parameter] = {name: Parameter(**kwargs) for name, kwargs in parameters.items()}

    def __getitem__(self, name: str) -> Any:  # noqa: ANN401
        """Gets a parameter by its name using `store[name]` syntax.

        Args:
            name (str): name of the parameter

        Returns:
            Any: the `value` of the parameter
        """
        return self.parameters[name].value

    def update(self) -> None:
        """🆙 updates every parameter."""
        for param in self.parameters.values():
            param.update()
