"""Timer — fires an async callback at configurable trigger points.

Supports pause/resume semantics so that a single cycle can survive
intermediate activity (e.g. model speaking a nudge response) without
resetting the trigger progress.

Set ``repeat_triggers=False`` to make each trigger fire at most once
per Timer instance. ``reset()`` still clears elapsed time and progress
but preserves the fired-trigger set, so exhausted triggers stay
exhausted across restarts.

    start()  - Begin from zero or resume from a paused state.
    stop()   - Pause, preserving elapsed time and trigger progress.
    reset()  - Clear elapsed time and progress to zero. Does not start.
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
        repeat_triggers: bool = True,
    ):
        self._triggers = sorted(triggers)
        self._on_trigger = on_trigger
        self._repeat_triggers = repeat_triggers
        self._fired_triggers: set[int] = set()
        self._task: Optional[asyncio.Task] = None
        self._run_start: Optional[float] = None
        self._elapsed_at_stop: float = 0.0
        self._trigger_index: int = 0
        self.is_running: bool = False

    def start(self) -> None:
        """Begin from zero or resume from a paused state.

        No-op if already running.
        """
        if self.is_running:
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

        Does not start the timer. Does **not** clear fired-trigger state
        when ``repeat_triggers=False``; construct a fresh ``Timer`` if a
        full reset is needed.
        """
        self._cancel_task()
        self._run_start = None
        self._elapsed_at_stop = 0.0
        self._trigger_index = 0
        self.is_running = False
        logger.debug("[Timer] Reset")

    async def _run(self) -> None:
        my_task = asyncio.current_task()
        try:
            while self._trigger_index < len(self._triggers):
                target = self._triggers[self._trigger_index]
                if not self._repeat_triggers and target in self._fired_triggers:
                    self._trigger_index += 1
                    continue
                wait = target - (time.monotonic() - self._run_start)
                if wait > 0:
                    await asyncio.sleep(wait)
                if not self.is_running or self._task is not my_task:
                    return
                logger.info("[Timer] Trigger fired at %ds", target)
                try:
                    await self._on_trigger(target)
                except Exception:
                    logger.exception(
                        "[Timer] on_trigger callback raised at t=%ds; continuing",
                        target,
                    )
                self._fired_triggers.add(target)
                self._trigger_index += 1
            if self._task is my_task:
                self._trigger_index = 0
                self._elapsed_at_stop = 0.0
        finally:
            # Only clear state if we are still the active task. A concurrent
            # reset()+start() may have replaced us with a new task that has
            # already claimed is_running / _task.
            if self._task is my_task:
                self.is_running = False
                self._task = None

    def _cancel_task(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None
