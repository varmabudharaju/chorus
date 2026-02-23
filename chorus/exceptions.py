"""Chorus exception hierarchy for clean error handling."""


class ChorusError(Exception):
    """Base exception for all Chorus errors."""


class ServerUnreachableError(ChorusError):
    """Cannot connect to the Chorus server."""


class SubmissionError(ChorusError):
    """Delta submission failed (409 duplicate, 400 bad file, etc)."""


class AggregationPendingError(ChorusError):
    """No aggregated model available yet."""


class RoundClosedError(ChorusError):
    """Round is not accepting submissions."""


class DuplicateClientError(ChorusError):
    """Client has already submitted to this round."""


class ExportError(ChorusError):
    """Failed to export merged model."""
