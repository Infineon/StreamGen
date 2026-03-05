# 🕰️ Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This project uses [*towncrier*](https://towncrier.readthedocs.io/) to build the changelog you are currently reading.

<!-- towncrier release notes start -->

## [2.0.0](https://github.com/Infineon/StreamGen/tree/2.0.0) - 2026-03-05

### ➖ Removed

- 🧹 removed pip extras to keep the package leaner ([#2](https://github.com/Infineon/StreamGen/issues/2))


## [1.2.0](https://github.com/Infineon/StreamGen/tree/1.2.0) - 2025-10-20

### ✨ Changed

- 🎲 made `rng` generators part of the `traverse` protocol and `SamplingTree`s. ([#12](https://github.com/Infineon/StreamGen/issues/12))


## [1.1.0](https://github.com/Infineon/StreamGen/tree/1.1.0) - 2025-01-08

### ➕ Added

- 📲 implemented `__setitem__` method for ParameterStore ([#9](https://github.com/Infineon/StreamGen/issues/9))
- 🤘 add additional shorthand initialization of a `ParameterStore` for Parameters without a schedule. ([#10](https://github.com/Infineon/StreamGen/issues/10))
- 🔮 global scope parameter passing logic. ([#11](https://github.com/Infineon/StreamGen/issues/11))


## [1.0.6](https://github.com/Infineon/StreamGen/tree/1.0.6) - 2025-01-08

### 📦 Misc

- 🎵🎶 migrated to poetry 2.0.0 ([#8](https://github.com/Infineon/StreamGen/issues/8))


## [1.0.4](https://github.com/Infineon/StreamGen/tree/1.0.4) - 2024-12-03

### 📦 Misc

- 📦 lowered anytree version constraint to >=2.7 ([#7](https://github.com/Infineon/StreamGen/issues/7))


## [1.0.3](https://github.com/Infineon/StreamGen/tree/1.0.3) - 2024-08-29

### ➖ Removed

- 🐍 dropped support for python 3.10 ([#4](https://github.com/Infineon/StreamGen/issues/4))

### 📦 Misc

- 🤖 added CI github actions ([#3](https://github.com/Infineon/StreamGen/issues/3))


## [1.0.1](https://github.com/Infineon/StreamGen/tree/1.0.1) - 2024-08-19

### ➖ Removed

- 🔥 temporarily removed the avalanche-lib stream construction helper functions due to an import problem in avalanche. ([#2](https://github.com/Infineon/StreamGen/issues/2))


## [1.0.0](https://github.com/Infineon/StreamGen/tree/1.0.0) - 2024-08-19

### 📦 Misc

- 1.0.0 release repo preparations 🎉 ([#1](https://github.com/Infineon/StreamGen/issues/1))
