"""Synthesizes model voice activity start/stop events from audio chunk timing.

The Gemini Live API only provides voice activity signals for the *user*.
This detector derives equivalent start/stop events for the *model* by
tracking when audio chunks arrive and calculating the expected playback
duration so that the "stop" event fires when the user would actually
hear the last sample.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional

from google.genai import types

from .models import Role, VoiceActivityData

logger = logging.getLogger(__name__)

BYTES_PER_SAMPLE = 2  # PCM16


class ModelVoiceActivityDetector:
    """
    Emits ``VoiceActivityData(role=MODEL)`` start/stop events.
    """

    def __init__(
        self,
        on_event: Callable[[VoiceActivityData], Awaitable[None]],
    ):
        self._on_event = on_event
        self._is_speaking = False
        self._turn_start: Optional[float] = None
        self._turn_audio_duration: float = 0.0
        self._stop_task: Optional[asyncio.Task] = None

    async def on_audio_chunk(self, data: bytes, sample_rate: int) -> None:
        """Called by the orchestrator for every model audio chunk."""
        if not self._is_speaking:
            self._is_speaking = True
            self._turn_start = time.monotonic()
            self._turn_audio_duration = 0.0
            self._cancel_stop_task()
            await self._on_event(VoiceActivityData(
                role=Role.MODEL,
                voice_activity_type=types.VoiceActivityType.ACTIVITY_START,
            ))

        self._turn_audio_duration += len(data) / (sample_rate * BYTES_PER_SAMPLE)

    async def on_turn_complete(self) -> None:
        """Called by the orchestrator on TURN_COMPLETE."""
        if not self._is_speaking:
            return

        elapsed = time.monotonic() - self._turn_start
        remaining = self._turn_audio_duration - elapsed
        self._cancel_stop_task()
        self._stop_task = asyncio.create_task(self._delayed_stop(max(0.0, remaining)))

    async def force_stop(self) -> None:
        """Called by the orchestrator on INTERRUPTED — emit END immediately."""
        self._cancel_stop_task()
        if self._is_speaking:
            self._is_speaking = False
            self._turn_start = None
            self._turn_audio_duration = 0.0
            await self._on_event(VoiceActivityData(
                role=Role.MODEL,
                voice_activity_type=types.VoiceActivityType.ACTIVITY_END,
            ))

    def cleanup(self) -> None:
        """Called on orchestrator shutdown — cancel pending tasks, no event."""
        self._cancel_stop_task()
        self._is_speaking = False
        self._turn_start = None
        self._turn_audio_duration = 0.0

    async def _delayed_stop(self, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            if self._is_speaking:
                self._is_speaking = False
                await self._on_event(VoiceActivityData(
                    role=Role.MODEL,
                    voice_activity_type=types.VoiceActivityType.ACTIVITY_END,
                ))
        except asyncio.CancelledError:
            return
        finally:
            self._stop_task = None
            self._turn_start = None
            self._turn_audio_duration = 0.0

    def _cancel_stop_task(self) -> None:
        if self._stop_task is not None and not self._stop_task.done():
            self._stop_task.cancel()
        self._stop_task = None
