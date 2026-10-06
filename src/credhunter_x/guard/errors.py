from __future__ import annotations


class LeakError(RuntimeError):
    """Raised when an outgoing payload contains part of a known secret, or the
    guard can't be sure it doesn't. Fail-closed: the call is always aborted."""
