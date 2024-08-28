"""🗃️ parameter stores are dictionary-like collections of parameters and schedules."""

from typing import Any, Self

import pandas as pd
from beartype import beartype
from loguru import logger
from rich.pretty import pretty_repr

from streamgen.parameter import Parameter, ScopedParameterDict


@beartype()
class ParameterStore:
    """🗃️ a dictionary-like container of `streamgen.parameter.Parameter` with their names as keys.

    The dictionary can be nested one level to create parameter scopes.
    All top-level parameters are considered as having `scope=None`.

    Args:
        parameters (list[Parameter] | ScopedParameterDict): parameters to store

    Raises:
        ValueError: if `parameters` are of type `ScopedParameterDict` and are nested more than two levels.

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
                    "scope1": {
                        "var1": { # var1 can be used again since its inside a scope
                            "value": 1,
                            "schedule": [2,3],
                            "strategy": "cycle",
                        },
                    },
                }
            )

        >>> store = ParameterStore([
                    Parameter("var1", 1, [2]),
                    Parameter("var2", schedule=[0.0, 1.0]),
                ]
            })
    """

    def __init__(self, parameters: list[Parameter] | ScopedParameterDict) -> None:  # noqa: D107
        self.scopes: set[str] = set()
        match parameters:
            case list(parameters):
                self.parameters: dict[str, Parameter] = {p.name: p for p in parameters}
                self.parameter_names: set[str] = {p.name for p in parameters}

            case dict(parameters):
                self.parameters = {}
                self.parameter_names: set[str] = set()

                for key, dictionary in parameters.items():
                    if self._dict_depth(dictionary) == 2:  # then key is a scope name  # noqa: PLR2004
                        scope = key
                        self.scopes.add(scope)
                        self.parameters[scope] = {}
                        # remove `name` entries since they are redundant
                        for name, parameter_kwargs in parameters[scope].items():
                            parameter_kwargs.pop("name", None)
                            self.parameters[scope][name] = Parameter(name=name, **parameter_kwargs)
                            self.parameter_names.add(f"{scope}.{self.parameters[scope][name].name}")
                    elif self._dict_depth(dictionary) == 1:  # otherwise its a top-level parameter
                        parameters[key].pop("name", None)
                        self.parameters[key] = Parameter(name=key, **parameters[key])
                        self.parameter_names.add(self.parameters[key].name)
                    else:
                        logger.warning("📚 parameters of type `ScopedParameterDict` should not be nested more than two levels.")
                        raise ValueError

    @staticmethod
    def _dict_depth(dictionary: Any) -> int:  # noqa: ANN401
        """📚 get the nesting-level/depth of a dictionary.

        Code adapted from https://www.geeksforgeeks.org/python-find-depth-of-a-dictionary/

        Args:
            dictionary (Any): a dictionary or any of its values

        Returns:
            int: depth of dictionary

        Examples:
            >>> store = ParameterStore._dict_depth({1: "a", 2: {3: {4: {}}}})
            4
        """
        if isinstance(dictionary, dict):
            return 1 + (max(map(ParameterStore._dict_depth, dictionary.values())) if dictionary else 0)
        return 0

    def __getitem__(self, name: str) -> Parameter:
        """🫱 gets a parameter by its name using `store[name]` syntax.

        Scoped parameters are fetched using `{scope}.{parameter.name}`.

        Args:
            name (str): name of the parameter

        Returns:
            Parameter: the parameter
        """
        if "." in name:
            scope, name = name.split(".")
            return self.parameters[scope][name]

        return self.parameters[name]

    def set_update_step(self, idx: int) -> None:
        """🕐 updates every parameter of `self` to a certain update step using `param[idx]`.

        Args:
            idx (int): parameter update step

        Returns:
            None: this function mutates `self`
        """
        for param in self.parameters.values():
            # indexing of parameters mutates their internal state, so we do not have to set anything
            param[idx]

    def get_scope(self, scope: str) -> Self | None:
        """🔭 get all parameters in a scope as a new `ParameterStore`.

        Args:
            scope (str): scope name

        Returns:
            ParameterStore | None: parameter store of all parameters inside the scope. None if scope is not present.
        """
        if scope not in self.scopes:
            return None
        return ParameterStore(list(self.parameters[scope].values()))

    def get_params(self) -> dict[str, Any]:
        """📖 get the current parameters as one dictionary.

        Scoped parameters are represented by "{scope}.{parameter.name}".

        Returns:
            dict[str, Any]: dictionary with scoped parameter names as keys and `parameter.value` as values.
        """
        return {name: self[name].value for name in sorted(self.parameter_names)}

    def update(self) -> dict[str, Any]:
        """🆙 updates every parameter and returns them using `ParameterStore.get_params`.

        Returns:
            dict[str, Any]: dictionary with `parameter.name` as keys and `parameter.value` as values
        """
        for name in self.parameter_names:
            self[name].update()

        return self.get_params()

    def to_dataframe(self, number_of_update_steps: int = 0) -> pd.DataFrame:
        """📅🐼 rolls out `number_of_update_steps` and returns the result as a `pd.DataFrame`.

        Args:
            number_of_update_steps (int, optional): number of rows in the dataframe. Defaults to 0.

        Returns:
            pd.DataFrame: parameter store as dataframe
        """
        return pd.DataFrame(
            [self.get_params()] + ([self.update() for _ in range(number_of_update_steps)] if number_of_update_steps > 0 else []),
        )

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> Self:
        """📅🐼 constructs a `ParameterStore` from a dataframe.

        Each columns represents a parameter.
        columns can be namespaced following the same "{scope}.{parameter.name}" rules as `ParameterStore.__getitem__`.

        Args:
            df (pd.DataFrame): dataframe

        Returns:
            ParameterStore: parameter store
        """
        scopes: set[str] = {name.split(".")[0] for name in df.columns if "." in name}

        params = {}
        for scope in scopes:
            params[scope] = {}
            for name in df.columns:
                if not name.startswith(f"{scope}."):
                    continue
                parameter_name = name.split(".")[1]
                params[scope][parameter_name] = {"schedule": df[name]}

        for name in df.columns:
            if "." in name:
                continue
            params[name] = {"schedule": df[name]}

        return ParameterStore(params)

    def __or__(self, params: Parameter | Self) -> Self:
        """➕ combines self with another `Parameter` or `ParameterStore` using `|`.

        This function takes care of merging the scopes properly.

        Args:
            params (Parameter | ParameterStore): another parameter or store

        Returns:
            ParameterStore: combined parameter store
        """  # noqa: RUF002
        if isinstance(params, Parameter):
            params = ParameterStore([params])

        for scope in params.scopes:
            if scope not in self.scopes:
                self.parameters[scope] = {}
                self.scopes.add(scope)

        for param in params.parameter_names:
            if "." in param:
                scope, name = param.split(".")
                self.parameters[scope][name] = params[param]
            else:
                self.parameters[param] = params[param]
            self.parameter_names.add(param)

        return self

    def __repr__(self) -> str:
        """🏷️ Returns the debug string representation of self.

        Returns:
            str: string representation of self
        """
        s = "("
        for name in sorted(self.parameter_names):
            s += str(self[name]) + ", "
        return s[:-2] + ")"

    def __str__(self) -> str:
        """🏷️ Returns the string representation `str(self)`.

        Returns:
            str: string representation of self
        """
        return pretty_repr(self.get_params())
