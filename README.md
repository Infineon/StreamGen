<p align="center">
    <img src="docs/images/stream_scene.png" alt="Banner"/></a>
</p>

<h1 style="text-align: center;">
    <img src="docs/images/milky-way_animated.png" style="display:inline; height:1.0em">
    StreamGen
</h1>

<p align="center">
a 🐍 framework for generating labeled data streams
</p>

<p align="center">
    <img alt="Static Badge" src="https://img.shields.io/badge/📦_version-0.0.1-blue">
    <a href="https://www.repostatus.org/#wip"><img src="https://www.repostatus.org/badges/latest/wip.svg" alt="Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public." /></a>
    <img alt="Static Badge" src="https://img.shields.io/badge/tests-passing-green?logo=pytest">
    <img alt="Static Badge" src="https://img.shields.io/badge/Coverage-65%25-yellow?logo=codecov">
</p>

<p align="center">
    <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python%203.11-darkgreen?style=for-the-badge&logo=python&logoColor=white"></a>
    <a href="https://python-poetry.org/"><img alt="Poetry" src="https://img.shields.io/badge/-Poetry%201.7-60A5FA?style=for-the-badge&logo=Poetry&logoColor=FFFFFF"></a>
</p>

<p align="center">
    <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
    <a href="https://github.com/beartype/beartype"><img alt="Beartype" src="https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg"></a>
</p>

---

## 📃 Table of Contents

- [💡 Idea](#💡-idea)
- [📦 Installation](#📦-installation)
- [👀 Examples](#👀-examples)
- [📖 Documentation](#📖-documentation)
- [🙏 Acknowledgement](#🙏-acknowledgement)

---

## 💡 Idea

![sampling tree](docs/images/sampling_tree.png)

![parameter schedule](docs/images/parameter_schedule.png)

![covariate shift](docs/images/covariate_shift.png)

![prior probability shift](docs/images/prior_probability_shift.png)

![concept shift](docs/images/concept_shift.png)

## 📦 Installation

```sh
pip install streamgen
```

## 👀 Examples

There are example notebooks 🪐📓 showcasing and explaining `streamgen` features:

+ 📈 time series
    + [🎲 sampling from static distributions](examples/time%20series%20classification/01_static_distributions.ipynb)
    + [🌌 creating data streams](examples/time%20series%20classification/02_data_streams.ipynb)
    + [📊 Data drift scenarios](examples/time%20series%20classification/03_drift_scenarios.ipynb)
+ 🖼️ analog wafer map streams based on the [wm811k dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) [^1] in [🌐 wafer map generation](examples/wafer_map_generation.ipynb)

## 📖 Documentation

Open the documentation by calling `poe docs`.

## 🙏 Acknowledgement

Made with ❤️ and ☕ by Laurenz Farthofer.

Special thanks to Benjamin Steinwender, Marius Birkenbach and Nikolaus Neugebauer for their valuable feedback.

I want to thank Infineon and Kai for letting me work on and publish this project.

Finally, I want to thank my university supervisors Thomas Pock and Marc Masana for their guidance.

---

## 📄 References

[^1]: Wu, Ming-Ju, Jyh-Shing R. Jang, and Jui-Long Chen. “Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets.” IEEE Transactions on Semiconductor Manufacturing 28, no. 1 (February 2015): 1–12.
