"""Orchestrator — wires transport, Gemini session, and tool handler into concurrent pipelines."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, Callable, Awaitable

from google.genai import types

from .models import (
    AudioData,
    TextData,
    EventData,
    Role,
    VoiceActivityData,
    TurnCompleteData,
)
from .transports.base_transport import BaseTransport
from .gemini_live_session import GeminiLiveSession, GeminiLiveResponseType
from .base_tool_handler import BaseToolHandler, ToolResponseAction
from .transcription import Transcription
from .metric_tracker import MetricTracker
from .timer import Timer
from .audio_recorder import AudioRecorder
from .model_voice_activity_detector import ModelVoiceActivityDetector

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorCallbacks:
    """Optional hooks for application-level reactions to orchestrator events.

    All callbacks are async. Unset callbacks (None) are silently skipped.
    """

    on_event: Optional[Callable[[EventData], Awaitable[None]]] = None
    on_turn_complete: Optional[
        Callable[[Optional[TurnCompleteData]], Awaitable[None]]
    ] = None
    on_interrupted: Optional[Callable[[], Awaitable[None]]] = None
    on_voice_activity: Optional[Callable[[VoiceActivityData], Awaitable[None]]] = None


class Orchestrator:
    """Orchestrator for Gemini Live Framework."""

    def __init__(
        self,
        transport: BaseTransport,
        gemini_session: GeminiLiveSession,
        tool_handler: Optional[BaseToolHandler] = None,
        audio_recorder: Optional[AudioRecorder] = None,
        transcription: Optional[Transcription] = None,
        callbacks: Optional[OrchestratorCallbacks] = None,
        user_idle_timer: Optional[Timer] = None,
        model_idle_timer: Optional[Timer] = None,
    ):
        self.transport = transport
        self.gemini_session = gemini_session
        self.tool_handler = tool_handler
        self.audio_recorder = audio_recorder
        self.transcription = transcription
        self.callbacks = callbacks or OrchestratorCallbacks()
        self._user_idle_timer = user_idle_timer
        self._model_idle_timer = model_idle_timer
        self.metric_tracker = MetricTracker()
        self._model_vad = ModelVoiceActivityDetector(
            on_event=self._on_model_voice_activity,
        )

    async def start(self) -> bool:
        """Start transport, connect Gemini in parallel, then run pipelines.

        Returns False if Gemini connection fails.
        """
        await self.transport.start()

        transport_task = asyncio.create_task(self._handle_transport_messages())

        connected = await self.gemini_session.connect()
        if not connected:
            transport_task.cancel()
            try:
                await transport_task
            except asyncio.CancelledError:
                pass
            await self.transport.stop()
            return False

        self.metric_tracker.start()

        if self.audio_recorder:
            self.audio_recorder.start()

        gemini_task = asyncio.create_task(self._handle_gemini_responses())

        await self._run_pipelines(transport_task, gemini_task)
        return True

    async def stop(self):
        self.metric_tracker.stop()
        if self._user_idle_timer:
            self._user_idle_timer.reset()
        if self._model_idle_timer:
            self._model_idle_timer.reset()
        await self.transport.stop()
        self._model_vad.cleanup()
        if self.tool_handler:
            await self.tool_handler.cleanup()
        await self.gemini_session.disconnect()
        if self.audio_recorder:
            asyncio.get_running_loop().run_in_executor(None, self.audio_recorder.stop)

    async def _handle_transport_messages(self):
        """Pipeline: transport -> Gemini."""
        async for message in self.transport.receive_message():
            match message:
                case AudioData() as audio:
                    self.metric_tracker.on_audio_sent()
                    if self.audio_recorder:
                        self.audio_recorder.record_user_audio(audio.data, audio.sample_rate)
                    await self.gemini_session.send_audio(audio.data)
                case TextData() as text:
                    await self.gemini_session.send_text(text.text)
                case EventData() as event:
                    if self.callbacks.on_event:
                        await self.callbacks.on_event(event)
        logger.info("[Orchestrator] Transport loop ended")

    async def _handle_gemini_responses(self):
        """Pipeline: Gemini -> transport (+ tool handler)."""
        async for response in self.gemini_session.receive_responses():
            match response.type:
                case GeminiLiveResponseType.AUDIO:
                    self.metric_tracker.on_audio_received()
                    if self.audio_recorder:
                        self.audio_recorder.record_model_audio(
                            response.data.data, response.data.sample_rate
                        )
                    await self.transport.send_audio(response.data)
                    await self._model_vad.on_audio_chunk(
                        response.data.data, response.data.sample_rate
                    )
                case GeminiLiveResponseType.TEXT:
                    await self.transport.send_text(response.data)
                case GeminiLiveResponseType.TRANSCRIPT:
                    await self.transport.send_transcript(response.data)
                    if response.data.role == Role.USER:
                        self.metric_tracker.on_user_transcript(response.data.text)
                    elif response.data.role == Role.MODEL:
                        self.metric_tracker.on_model_transcript(response.data.text)
                    if self.transcription:
                        if response.data.role == Role.USER:
                            await self.transcription.on_user_transcript(response.data.text)
                        elif response.data.role == Role.MODEL:
                            await self.transcription.on_model_transcript(response.data.text)
                case GeminiLiveResponseType.INTERRUPTED:
                    self.metric_tracker.on_interruption()
                    await self._model_vad.force_stop()
                    if self.transcription:
                        await self.transcription.on_interrupted()
                    await self.transport.send_interruption()
                    if self.callbacks.on_interrupted:
                        await self.callbacks.on_interrupted()
                case GeminiLiveResponseType.VOICE_ACTIVITY:
                    logger.info(f"[Orchestrator] Voice activity: User -> {response.data.voice_activity_type.value}")
                    if response.data.voice_activity_type == types.VoiceActivityType.ACTIVITY_START:
                        self.metric_tracker.on_user_turn()
                        if self._user_idle_timer:
                            self._user_idle_timer.reset()
                        if self._model_idle_timer:
                            self._model_idle_timer.reset()
                    elif response.data.voice_activity_type == types.VoiceActivityType.ACTIVITY_END:
                        if self._model_idle_timer:
                            self._model_idle_timer.start()
                    await self.transport.send_voice_activity(response.data)
                    if self.callbacks.on_voice_activity:
                        await self.callbacks.on_voice_activity(response.data)
                case GeminiLiveResponseType.TURN_COMPLETE:
                    await self._model_vad.on_turn_complete()
                    if self.transcription:
                        await self.transcription.on_model_turn_complete()
                    if self.callbacks.on_turn_complete:
                        tc = (
                            response.data
                            if isinstance(response.data, TurnCompleteData)
                            else TurnCompleteData(audio_chunks=0)
                        )
                        await self.callbacks.on_turn_complete(tc)
                case GeminiLiveResponseType.TOOL_CALL:
                    self.metric_tracker.on_tool_call()
                    if self.tool_handler:
                        await self.tool_handler.handle_tool_call(response.data)
                    else:
                        logger.warning("[Orchestrator] Tool call received but no handler registered")
                case GeminiLiveResponseType.USAGE_METADATA:
                    self.metric_tracker.on_usage_metadata(response.data)
                case GeminiLiveResponseType.TOOL_CALL_CANCELLATION:
                    if self.tool_handler:
                        await self.tool_handler.handle_cancellation(response.data.ids)
                    else:
                        logger.warning("[Orchestrator] Tool cancellation received but no handler registered")
        logger.info("[Orchestrator] Gemini response loop ended")

    async def _handle_tool_results(self):
        """Pipeline: tool handler result queue -> Gemini."""
        while True:
            result = await self.tool_handler.result_queue.get()
            if result is None:
                break
            try:
                match result.action:
                    case ToolResponseAction.SEND_RESPONSE:
                        await self.gemini_session.send_tool_response(
                            result.tool_id, result.tool_name, result.result,
                        )
                    case ToolResponseAction.SEND_INTERIM:
                        await self.gemini_session.send_interim_tool_response(
                            result.tool_id, result.tool_name, result.interim_message,
                        )
                    case ToolResponseAction.SEND_CONTEXT:
                        await self.gemini_session.send_tool_result_as_context(
                            result.tool_id, result.tool_name, result.result,
                        )
            except Exception as e:
                logger.error(f"[Orchestrator] Failed to forward tool result: {e}")

    async def _run_pipelines(
        self,
        transport_task: asyncio.Task,
        gemini_task: asyncio.Task,
    ):
        """Wait on the already-running transport and Gemini tasks.

        Uses FIRST_COMPLETED so that when either pipeline ends (e.g.
        WebSocket disconnect, server shutdown) the remaining tasks are
        cancelled promptly instead of hanging.
        """
        tasks = [transport_task, gemini_task]

        try:
            if self.tool_handler:
                tasks.append(asyncio.create_task(self._handle_tool_results()))

            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                exc = task.exception() if not task.cancelled() else None
                if exc:
                    logger.error("[Orchestrator] Pipeline failed: %s", exc)
        except Exception as e:
            logger.error(f"[Orchestrator] Processing loop failed: {e}")
            pending = {t for t in tasks if not t.done()}
        finally:
            for task in pending:
                task.cancel()
            for task in pending:
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            await self.stop()

    async def _on_model_voice_activity(self, data: VoiceActivityData) -> None:
        """Internal handler for synthesized model voice activity events."""
        logger.info(f"[Orchestrator] Voice activity: Model -> {data.voice_activity_type.value}")
        if data.voice_activity_type == types.VoiceActivityType.ACTIVITY_START:
            self.metric_tracker.on_model_turn()
            if self._model_idle_timer:
                self._model_idle_timer.reset()
            if self._user_idle_timer:
                self._user_idle_timer.stop()
        elif data.voice_activity_type == types.VoiceActivityType.ACTIVITY_END:
            if self._user_idle_timer:
                self._user_idle_timer.start()
        if self.callbacks.on_voice_activity:
            await self.callbacks.on_voice_activity(data)
