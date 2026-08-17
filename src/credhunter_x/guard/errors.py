from __future__ import annotations


class LeakError(RuntimeError):
    """Raised when an outgoing payload contains a fragment of a known real
    secret value, or when the guard cannot be sure it isn't missing one
    (e.g. an empty registry). The guard is fail-closed: this must always
    abort the call — never be caught and logged-and-continued."""
