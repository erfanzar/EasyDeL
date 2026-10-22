"""Release frozen rollout policies only after their last asynchronous reader exits."""

from __future__ import annotations

import threading
import typing as tp
from concurrent.futures import Future


def after_futures(futures: tp.Iterable[Future], callback: tp.Callable[[], None]) -> None:
    """Invoke a callback once all readers finish, without joining a stuck worker.

    Register after submissions have stopped. Completed and cancelled futures
    count as finished; callbacks keep the policy owner alive until release.
    """
    pending = set(futures)
    lock = threading.Lock()
    if not pending:
        callback()
        return

    def finished(future: Future) -> None:
        nonlocal pending
        with lock:
            pending.discard(future)
            last = not pending
        if last:
            callback()

    for future in tuple(pending):
        future.add_done_callback(finished)
