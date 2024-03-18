# ⚒️ Developers Guide

Coding is a very free and expressive activity ❤️. Due to this freedom, it is extremely advantageous to adhere to some standards inside a team.

Generally, we try to adhere to the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
It is a long read, but worth your time if you want to improve your style and become more sure on which features to use (which is important since there are often many ways to do things).

Some notable deviations from the Google style guide:

+ we prefer a longer line length of 140 characters
+ we prefer *f-strings* over other string formatting techniques (this refers to [Google Python Style Guide - Logging](https://google.github.io/styleguide/pyguide.html#3101-logging))

---

## Logging

We use the [loguru](https://github.com/Delgan/loguru) logger library for logging messages and [rich](https://github.com/Textualize/rich) progress bars for longer running processes.

The `loguru` logger has a number of advantages over simple print statements:

+ log levels provide a nice visual feedback and help finding problems and warning in longer logs
+ log levels can be configured (if you do not want to see `logger.debug` messages, set the log-level to something higher like `INFO`, or `WARNING`)
+ log messages automatically include timestamps, module information and code lines. e.g.:
+ logs can be automatically saved to log files (even with configurable rotation - files don't grow infinitely )

The standard Python log-levels should be used in their intended way. Developers have to judge if information is only interesting during development or also for a user.

We prioritize logging before an action, to know in which step a program crashed in case of an error. One exception to this is when we want to give additional information after a process, in which case we prefer to only log after the process for short processes, or use multiple logging statements or progress bars. To avoid too verbose logs, we do not log "done" statements. They are implicit by either a termination, or a new pre-action logging statement.

Logging messages can be categorized into 2 groups:

1) describing actions,

    when something is happening after the logging statement. No termination dot is used at the end of the statement, but can be used in between the logs if it improves the clarity.

    ```
    <emoji> <lower-case verb in present continuous form ("-ing" form)> <description of what is happening>
    ```

2) giving information,

    to give additional information (like how many files were found or how much memory a dataframe consumes). These statements are either written in past tense (e.g. "📑 found 4 files in './data'"), or in present tense if an object is described (e.g. "💽 dataframe consumes 0.2 GB of heap space").

## Pre-commit hooks 🛑

[Pre-commit](https://pre-commit.com/) hooks are scripts that are executed when you do a `git commit`. They generally check for code quality and correctness and a lot of hooks can even automatically correct issues. Since they modify files, you have to `git add` and `git commit` them again.

To install the pre-commit hooks that are listed and configured in `.pre-commit-config.yaml` run:
```bash
poetry shell
pre-commit install
```

> Admin rights are not strictly needed to install the hooks, but we found that installing the hooks with admin rights can avoid access related errors

Sometimes, you might not have the time or patience to correct an error reported by a hook. In such cases, we strongly advise against removing/disabling pre-commit or the concrete hook. Often, hooks allow to ignore certain errors on a line basis by using special comments (e.g. [mypy - Silencing linter](https://mypy.readthedocs.io/en/stable/common_issues.html#silencing-linters)).

## Type annotations 🚀

Although Python is a dynamically-typed language, you can still annotate your code with type annotations.
These annotations will be ignored by the Python interpreter, but offer serious benefits:

+ static type checkers like [mypy](https://github.com/python/mypy) (included in our pre-commit hooks) can catch a lot of difficult errors for you
+ intentions of variables are clearer - no need to include type information in the names of your variables
+ function interfaces are documented directly with your code

Because `streamgen` is a library that is used by developers who might not use static type checkers like `mypy`, we use the dynamic type checker [Beartype](https://github.com/beartype/beartype), which checks the types of function arguments and return value at runtime. This avoids misuse of the functions without having to write type checks manually.

For an introduction on how to annotate types, have a look at the [Python typing docs](https://docs.python.org/3/library/typing.html).

## Docstrings 🏷️

By far the best way to document your code and functions is by using docstrings. There are several standard styles of writing docstring. We use the [Google-style](https://google.github.io/styleguide/pyguide.html#s3.8.1-comments-in-doc-strings).

## Task automation 🦾

[Poe the Poet](https://poethepoet.natn.io/) is a batteries included task runner that works well with Poetry.

It provides a simple way to define project tasks within your `pyproject.toml`, and either a standalone CLI or a Poetry plugin to run them using your project's virtual environment.
