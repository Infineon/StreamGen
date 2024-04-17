"""🌌 stream abstractions."""

from beartype import beartype

from streamgen import is_extra_installed

if is_extra_installed("cl"):
    pass


@beartype()
class Experience:  # noqa: D101
    pass


@beartype()
class Stream:  # noqa: D101
    pass


@beartype()
class SizedStream(Stream):  # noqa: D101
    pass
