"""Transcription service — maintains a merged conversation history and emits transcript callbacks."""

from __future__ import annotations

import datetime
import logging
from enum import Enum
from typing import Awaitable, Callable, Optional

from .models import Role, TranscriptData, TranscriptEntry

logger = logging.getLogger(__name__)


class TranscriptMode(Enum):
    MERGED = "merged"
    STREAMING = "streaming"


class Transcription:
    """
    Accumulates user and model transcripts into an interleaved conversation history.
    """

    def __init__(
        self,
        mode: TranscriptMode = TranscriptMode.MERGED,
        on_transcript: Optional[Callable[[TranscriptData], Awaitable[None]]] = None,
    ):
        self.mode = mode
        self._on_transcript = on_transcript
        self.entries: list[TranscriptEntry] = []
        self._model_buffer: str = ""

    async def on_user_transcript(self, text: str) -> None:
        """Called by the orchestrator when a user transcript chunk arrives."""
        if self.entries and self.entries[-1].role == Role.USER:
            self.entries[-1].text += text
            logger.info(f"[Transcription] User: {self.entries[-1].text}")
        else:
            logger.info(f"[Transcription] User: {text}")
            self.entries.append(TranscriptEntry(
                role=Role.USER,
                text=text,
                timestamp=datetime.datetime.now(datetime.timezone.utc),
            ))
        if self._on_transcript:
            await self._on_transcript(TranscriptData(role=Role.USER, text=text))

    async def on_model_transcript(self, text: str) -> None:
        """Called by the orchestrator when a model transcript chunk arrives."""
        self._model_buffer += text

        if self.mode == TranscriptMode.STREAMING and self._on_transcript:
            await self._on_transcript(TranscriptData(
                role=Role.MODEL, text=self._model_buffer,
            ))

    async def on_model_turn_complete(self) -> None:
        """Called by the orchestrator on TURN_COMPLETE."""
        await self._finalize_model(interrupted=False)

    async def on_interrupted(self) -> None:
        """Called by the orchestrator on INTERRUPTED."""
        await self._finalize_model(interrupted=True)

    async def _finalize_model(self, interrupted: bool) -> None:
        if not self._model_buffer:
            return

        logger.info(f"[Transcription] Model: {self._model_buffer}")
        self.entries.append(TranscriptEntry(
            role=Role.MODEL,
            text=self._model_buffer,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            interrupted=interrupted,
        ))

        if self.mode == TranscriptMode.MERGED and self._on_transcript:
            await self._on_transcript(TranscriptData(
                role=Role.MODEL, text=self._model_buffer,
            ))

        self._model_buffer = ""
