"""🪐 jupyter configurations."""  # noqa: INP001"

c = get_config()  # noqa: F821
c.NbConvertApp.notebooks = [
    "examples/time series classification/01_static_distributions.ipynb",
    "examples/time series classification/02_data_streams.ipynb",
    "examples/time series classification/03_drift_scenarios.ipynb",
    "examples/wafer_map_generation.ipynb",
]
