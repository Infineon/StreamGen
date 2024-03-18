# 🗂️ Project Structure

+ 📖 `docs/` - contains the documentation you are currently reading
+ 👀 `examples/` - jupyter notebooks showcasing how to use the library
+ ⚗️ `tests/` - pytest test suite
+ 🧑‍💻 `streamgen/` - source code
    + `parameter/` - ⚙️ parameters are variables that change over time according to a schedule
        + `store.py` - 🗃️ parameter stores are dictionary-like collections of parameters and schedules
    + `samplers/` - 🎲 implementations for different samplers
        + `tree.py` - 🌳 sampling trees are trees of transformations that you can traverse from root to leaf to create samples
    + `enums.py` - 🔢 all enumerations for `streamgen`
    + `exceptions.py` - 🔥 module containing `streamgen` exceptions
    + `nodes.py` - 🪢 different node implementations using [anytree](https://anytree.readthedocs.io/en/stable/) `NodeMixin`
    + `streams.py` - 🌌 stream abstractions
    + `transforms.py` - ➡️ useful transformations
+ 🐍 `pyproject.toml` - project configuration file
+ 📄 `README.md` - documentation entry point
