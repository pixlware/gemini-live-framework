"""Timer — fires an async callback at configurable trigger points.

Supports pause/resume semantics so that a single cycle can survive
intermediate activity (e.g. model speaking a nudge response) without
resetting the trigger progress.

    start()  - Begin from zero or resume from a paused state.
    stop()   - Pause, preserving elapsed time and trigger progress.
    reset()  - Clear elapsed time and progress to zero. Does not start.
    end()    - Force-stop and count the current run as a completed cycle.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


class Timer:
    """Async timer that sleeps to sorted trigger points and fires a callback."""

    def __init__(
        self,
        triggers: list[int],
        on_trigger: Callable[[int], Awaitable[None]],
        max_cycles: Optional[int] = None,
    ):
        self._triggers = sorted(triggers)
        self._on_trigger = on_trigger
        self._max_cycles = max_cycles
        self._task: Optional[asyncio.Task] = None
        self._run_start: Optional[float] = None
        self._elapsed_at_stop: float = 0.0
        self._trigger_index: int = 0
        self.is_running: bool = False
        self.cycles_completed: int = 0

    def start(self) -> None:
        """Begin from zero or resume from a paused state.

        No-op if already running or ``max_cycles`` has been reached.
        """
        if self.is_running:
            return
        if self._max_cycles is not None and self.cycles_completed >= self._max_cycles:
            logger.debug(
                "[Timer] Start ignored — max_cycles reached (%d/%d)",
                self.cycles_completed,
                self._max_cycles,
            )
            return
        self._cancel_task()
        if self._elapsed_at_stop > 0:
            self._run_start = time.monotonic() - self._elapsed_at_stop
        else:
            self._run_start = time.monotonic()
        self.is_running = True
        self._task = asyncio.create_task(self._run())
        logger.debug(
            "[Timer] Started (trigger_index=%d, elapsed=%.1fs)",
            self._trigger_index,
            self._elapsed_at_stop,
        )

    def stop(self) -> None:
        """Pause the timer, preserving elapsed time and trigger progress."""
        if not self.is_running:
            return
        self._elapsed_at_stop = time.monotonic() - self._run_start
        self._cancel_task()
        self.is_running = False
        logger.debug(
            "[Timer] Stopped (elapsed=%.1fs, trigger_index=%d)",
            self._elapsed_at_stop,
            self._trigger_index,
        )

    def reset(self) -> None:
        """Clear elapsed time and trigger progress to zero.

        Does not start the timer.  Does not affect ``cycles_completed``.
        """
        self._cancel_task()
        self._run_start = None
        self._elapsed_at_stop = 0.0
        self._trigger_index = 0
        self.is_running = False
        logger.debug("[Timer] Reset")

    def end(self) -> None:
        """Force-stop and count the current run as a completed cycle."""
        self._cancel_task()
        self._run_start = None
        self._elapsed_at_stop = 0.0
        self._trigger_index = 0
        self.is_running = False
        self.cycles_completed += 1
        logger.debug("[Timer] Ended (cycles_completed=%d)", self.cycles_completed)

    @property
    def is_active(self) -> bool:
        """True if the timer task is still pending."""
        return self._task is not None and not self._task.done()

    async def _run(self) -> None:
        try:
            while self._trigger_index < len(self._triggers):
                target = self._triggers[self._trigger_index]
                wait = target - (time.monotonic() - self._run_start)
                if wait > 0:
                    await asyncio.sleep(wait)
                logger.debug("[Timer] Trigger fired at %ds", target)
                await self._on_trigger(target)
                self._trigger_index += 1
            self.cycles_completed += 1
            self._trigger_index = 0
            self._elapsed_at_stop = 0.0
        except asyncio.CancelledError:
            return
        finally:
            self.is_running = False
            self._task = None

    def _cancel_task(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None
