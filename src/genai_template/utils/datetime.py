from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return the current UTC time."""

    return datetime.now(UTC)
