"""Transcription service — maintains a merged conversation history and emits transcript callbacks."""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Awaitable, Callable, Optional

from .models import Role, TranscriptData, TranscriptEntry
from .logger import SessionLogger


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
        logger: Optional[SessionLogger] = None,
    ):
        self.mode = mode
        self._on_transcript = on_transcript
        self.entries: list[TranscriptEntry] = []
        self._model_buffer: str = ""
        self._session_logger = logger

    @property
    def logger(self) -> SessionLogger:
        if self._session_logger is None:
            self._session_logger = SessionLogger()
        return self._session_logger

    @logger.setter
    def logger(self, logger: SessionLogger) -> None:
        self._session_logger = logger

    async def on_user_transcript(self, text: str) -> None:
        """Called by the orchestrator when a user transcript chunk arrives."""
        if self.entries and self.entries[-1].role == Role.USER:
            self.entries[-1].text += text
            self.logger.info(f"[Transcription] User: {self.entries[-1].text}")
        else:
            self.logger.info(f"[Transcription] User: {text}")
            self.entries.append(TranscriptEntry(
                role=Role.USER,
                text=text,
                timestamp=datetime.datetime.now(datetime.timezone.utc),
            ))
        await self._emit_transcript(TranscriptData(role=Role.USER, text=text))

    async def on_model_transcript(self, text: str) -> None:
        """Called by the orchestrator when a model transcript chunk arrives."""
        self._model_buffer += text

        if self.mode == TranscriptMode.STREAMING:
            await self._emit_transcript(TranscriptData(role=Role.MODEL, text=text))

    async def on_model_turn_complete(self) -> None:
        """Called by the orchestrator on TURN_COMPLETE."""
        await self._finalize_model(interrupted=False)

    async def on_interrupted(self) -> None:
        """Called by the orchestrator on INTERRUPTED."""
        await self._finalize_model(interrupted=True)

    async def _finalize_model(self, interrupted: bool) -> None:
        if not self._model_buffer:
            return

        final_text = self._model_buffer
        self._model_buffer = ""

        self.logger.info(f"[Transcription] Model: {final_text}")
        self.entries.append(TranscriptEntry(
            role=Role.MODEL,
            text=final_text,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            interrupted=interrupted,
        ))

        payload_text = final_text if self.mode == TranscriptMode.MERGED else ""
        await self._emit_transcript(TranscriptData(
            role=Role.MODEL,
            text=payload_text,
            final=True,
            interrupted=interrupted,
        ))

    async def _emit_transcript(self, data: TranscriptData) -> None:
        """Fire the consumer's ``on_transcript`` callback if one is wired.

        A raising callback is logged and swallowed so a buggy consumer
        hook can't tear down a live call. The history entry has already
        been appended by the caller, so ``self.entries`` stays
        consistent even when the callback fails.
        """
        if not self._on_transcript:
            return
        try:
            await self._on_transcript(data)
        except Exception as e:
            self.logger.error(
                "[Transcription] on_transcript callback raised; continuing",
                error=str(e)
            )
