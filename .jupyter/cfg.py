"""🪐 jupyter configurations."""  # noqa: INP001"

c = get_config()  # noqa: F821
c.NbConvertApp.notebooks = [
    "examples/time series classification/01-static-distributions.ipynb",
    "examples/time series classification/02-data-streams.ipynb",
    "examples/time series classification/03-drift-scenarios.ipynb",
    "examples/time series classification/04-multi-label-generation.ipynb",
    "examples/wafer_map_generation.ipynb",
]
c.NbConvertApp.output_files_dir = "{notebook_name}-files"
