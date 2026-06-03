"""Synthesizes model voice activity start/stop events from audio chunk timing."""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, Optional

from google.genai import types

from .models import Role, VoiceActivityData

BYTES_PER_SAMPLE = 2  # PCM16
MODEL_VAD_END_DELAY = 0.3


class ModelVAD:
    """Emits ``VoiceActivityData(role=MODEL)`` start/stop events."""

    def __init__(
        self,
        on_event: Callable[[VoiceActivityData], Awaitable[None]],
    ):
        self._on_event = on_event
        self._is_speaking = False
        self._turn_start: float = 0.0
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
        remaining = self._turn_audio_duration - elapsed + MODEL_VAD_END_DELAY
        self._cancel_stop_task()
        self._stop_task = asyncio.create_task(self._delayed_stop(max(0.0, remaining)))

    async def force_stop(self) -> None:
        """Called by the orchestrator on INTERRUPTED — emit END immediately."""
        self._cancel_stop_task()
        if self._is_speaking:
            self._reset_turn_state()
            await self._on_event(VoiceActivityData(
                role=Role.MODEL,
                voice_activity_type=types.VoiceActivityType.ACTIVITY_END,
            ))

    def cleanup(self) -> None:
        """Called on orchestrator shutdown — cancel pending tasks, no event."""
        self._cancel_stop_task()
        self._reset_turn_state()

    async def _delayed_stop(self, delay: float) -> None:
        task = asyncio.current_task()
        try:
            await asyncio.sleep(delay)
            self._reset_turn_state()
            await self._on_event(VoiceActivityData(
                role=Role.MODEL,
                voice_activity_type=types.VoiceActivityType.ACTIVITY_END,
            ))
        finally:
            if self._stop_task is task:
                self._stop_task = None

    def _cancel_stop_task(self) -> None:
        if self._stop_task is not None and not self._stop_task.done():
            self._stop_task.cancel()
        self._stop_task = None

    def _reset_turn_state(self) -> None:
        self._is_speaking = False
        self._turn_start = 0.0
        self._turn_audio_duration = 0.0
