"""🌌 stream abstractions."""

from beartype import beartype


@beartype()
class Experience:
    pass


@beartype()
class Stream:
    pass


@beartype()
class SizedStream(Stream):
    pass
