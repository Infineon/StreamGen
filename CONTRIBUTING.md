# Contributing to StreamGen
:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

## Reporting Bugs
Before creating bug reports, please check existing issues list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible in the issue using the [🐛 bug report issue template](https://github.com/Infineon/StreamGen/blob/main/.github/ISSUE_TEMPLATE/%F0%9F%90%9B%20bug%20report.md).

## Suggesting Enhancements

For concrete ideas, open new issue using the [🚀 feature request template](https://github.com/Infineon/StreamGen/blob/main/.github/ISSUE_TEMPLATE/%F0%9F%9A%80%20feature%20request.md).

To start a or participate in a discussion, head over to https://github.com/Infineon/StreamGen/discussions

## Pull requests

Fill in the [pull request template](https://github.com/Infineon/StreamGen/blob/main/.github/pull_request_template.md) and make sure your code is documented.

## Setup development environment

### Requirements

* Poetry: [https://python-poetry.org/docs/](https://python-poetry.org/docs/)

After installing Poetry and cloning the project from GitHub, you should run the following command from the root of the cloned project:

```sh
poetry install --all-extras --sync --compile
```

All of the project's dependencies should be installed and the project ready for further development.

To activate the newly created virtual environment, call `poetry shell`.

## Development Tasks

StreamGen uses [Poe the Poet](https://github.com/nat-n/poethepoet) as a task runner, to document common workflows and simplify their invocation.

Poe tasks are defined in the `pyproject.toml` file.

### Testing

Manually run the tests:

```sh
poe test
```

### Documentation

Build the documentation locally:

```sh
poe docs_local
```

Build and publish docs:

```sh
poe docs
```
