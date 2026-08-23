from __future__ import annotations


class LeakError(RuntimeError):
    """Raised when an outgoing payload contains a fragment of a known
    secret, or the guard can't be sure it doesn't. Fail-closed: must
    always abort the call, never be caught and continued."""
