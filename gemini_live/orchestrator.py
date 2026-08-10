"""Orchestrator — wires transport, Gemini session, and tool handler into concurrent pipelines."""

import asyncio
from dataclasses import dataclass
from typing import Any, Optional, Callable, Awaitable

from google.genai import types

from .logger import SessionLogger

from .models import (
    AudioData,
    TextData,
    EventData,
    InterruptionData,
    Role,
    ToolCallCancellationData,
    ToolCallData,
    TranscriptData,
    TurnCompleteData,
    UsageMetadataData,
    VoiceActivityData,
)
from .transports.base_transport import BaseTransport
from .gemini_live_session import GeminiLiveSession, GeminiLiveResponseType
from .base_tool_handler import BaseToolHandler, ToolHandlerResult, ToolResponseAction
from .transcription import Transcription
from .metric_tracker import MetricTracker
from .timer import Timer
from .audio_recorder import AudioRecorder
from .model_vad import ModelVAD
from .turn_tracker import ConversationState, TurnTracker


@dataclass
class OrchestratorCallbacks:
    """Optional hooks for application-level reactions to orchestrator events.

    All callbacks are async. Unset callbacks (None) are silently skipped.
    """

    on_event: Optional[Callable[[EventData], Awaitable[None]]] = None
    on_turn_complete: Optional[
        Callable[[Optional[TurnCompleteData]], Awaitable[None]]
    ] = None
    on_interrupted: Optional[Callable[[InterruptionData], Awaitable[None]]] = None
    on_voice_activity: Optional[Callable[[VoiceActivityData], Awaitable[None]]] = None
    on_turn_state_change: Optional[
        Callable[[ConversationState, ConversationState, Optional[float]], Awaitable[None]]
    ] = None
    on_usage_metadata: Optional[
        Callable[[UsageMetadataData], Awaitable[None]]
    ] = None


class OrchestratorActions:
    """Unified actions manager for controlling orchestrator state and triggering transitions."""

    def __init__(self, turn_tracker: TurnTracker):
        self._turn_tracker = turn_tracker

    async def trigger_bot_timeout(self) -> None:
        """Force transition the turn tracker to WAITING_FOR_USER after a bot idle timeout."""
        await self._turn_tracker.bot_timeout()



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
        logger: Optional[Any] = None,
    ):
        self.logger = logger or SessionLogger()

        self.transport = transport
        self.transport.logger = self.logger

        self.gemini_session = gemini_session
        self.gemini_session.logger = self.logger

        self.tool_handler = tool_handler
        if self.tool_handler:
            self.tool_handler.logger = self.logger
            self.tool_handler.set_send_tool_result(self._send_tool_result)

        self.audio_recorder = audio_recorder
        if self.audio_recorder:
            self.audio_recorder.logger = self.logger

        self.transcription = transcription
        if self.transcription:
            self.transcription.logger = self.logger
        self.callbacks = callbacks or OrchestratorCallbacks()
        self.metric_tracker = MetricTracker(logger=self.logger)
        self._model_vad = ModelVAD(
            on_event=self._on_model_voice_activity,
        )
        self._turn_tracker = TurnTracker(
            user_idle_timer=user_idle_timer,
            model_idle_timer=model_idle_timer,
            on_state_change=self._on_turn_state_change,
            logger=self.logger,
        )
        self.actions = OrchestratorActions(self._turn_tracker)
        self._tool_dispatch_tasks: set[asyncio.Task] = set()
        self.is_running: bool = False

        # Batch accumulator for multi-tool turns: when the model emits 2+
        # blocking tool calls in a single turn, buffer their FunctionResponses
        # and send them in one batched send_tool_response call.
        self._current_turn_tool_ids: list[str] = []
        self._batch_tool_ids: set[str] = set()
        self._pending_batch: dict[str, ToolHandlerResult] = {}

        self._response_handlers: dict[
            GeminiLiveResponseType, Callable[[Any], Awaitable[None]]
        ] = {
            GeminiLiveResponseType.AUDIO: self._on_audio,
            GeminiLiveResponseType.TEXT: self._on_text,
            GeminiLiveResponseType.TRANSCRIPT: self._on_transcript,
            GeminiLiveResponseType.INTERRUPTED: self._on_interrupted,
            GeminiLiveResponseType.VOICE_ACTIVITY: self._on_user_voice_activity,
            GeminiLiveResponseType.TURN_COMPLETE: self._on_turn_complete,
            GeminiLiveResponseType.TOOL_CALL: self._on_tool_call,
            GeminiLiveResponseType.USAGE_METADATA: self._on_usage_metadata,
            GeminiLiveResponseType.TOOL_CALL_CANCELLATION: self._on_tool_cancellation,
        }

    @property
    def conversation_state(self) -> ConversationState:
        """Current conversation state as tracked by the turn tracker."""
        return self._turn_tracker.state

    async def start(self) -> bool:
        """Start transport, connect Gemini in parallel, then run pipelines.

        Returns False if Gemini connection fails.
        """
        if self.is_running:
            self.logger.warning("[Orchestrator] start() called while already running; ignoring")
            return True
        self.is_running = True

        try:
            await self.transport.start()

            transport_task = asyncio.create_task(self._handle_transport_messages())

            # Race the Gemini connect against the client transport so an early
            # client disconnect aborts startup instead of completing a full
            # connect + greeting + teardown for a client that is already gone.
            connect_task = asyncio.create_task(self.gemini_session.connect())
            done, _ = await asyncio.wait(
                {connect_task, transport_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if connect_task not in done:
                self.logger.info(
                    "[Orchestrator] Client disconnected before Gemini connected; aborting startup"
                )
                connect_task.cancel()
                try:
                    await connect_task
                except (asyncio.CancelledError, Exception):
                    pass
                await self.gemini_session.disconnect()
                await self.transport.stop()
                self.is_running = False
                return False

            connected = connect_task.result()
            if not connected:
                transport_task.cancel()
                try:
                    await transport_task
                except asyncio.CancelledError:
                    pass
                await self.transport.stop()
                self.is_running = False
                return False

            self.metric_tracker.start()

            if self.audio_recorder:
                self.audio_recorder.start()

            if not self.gemini_session.initial_text:
                await self._turn_tracker.await_user()

            gemini_task = asyncio.create_task(self._handle_gemini_responses())

            await self._run_pipelines(transport_task, gemini_task)
            return True
        except BaseException:
            self.is_running = False
            raise

    async def stop(self):
        if not self.is_running:
            return
        self.is_running = False

        # Phase 1 — shut down local state machines (cheap, sync).
        await self._turn_tracker.stop()
        self._model_vad.cleanup()

        # Phase 2 — cancel all tool work and drain in-flight dispatches so no
        # one tries to call gemini_session.send_tool_response against a closed
        # session below.
        for task in list(self._tool_dispatch_tasks):
            if not task.done():
                task.cancel()
        if self._tool_dispatch_tasks:
            await asyncio.gather(*self._tool_dispatch_tasks, return_exceptions=True)
        self._tool_dispatch_tasks.clear()
        self._current_turn_tool_ids.clear()
        self._batch_tool_ids.clear()
        self._pending_batch.clear()
        if self.tool_handler:
            await self.tool_handler.cleanup()

        # Phase 3 — close network sessions outside-in: Gemini first (source),
        # transport last (sink) so any final flush still has a live channel.
        await self.gemini_session.disconnect()
        await self.transport.stop()

        # Phase 4 — finalize recording. AudioRecorder.stop() never raises; it
        # logs its own errors and discards audio on failure.
        if self.audio_recorder:
            asyncio.get_running_loop().run_in_executor(
                None, self.audio_recorder.stop
            )

        # Phase 5 — freeze metrics last so end_time reflects true session end.
        self.metric_tracker.stop(log_summary=True)

    def _dispatch_tool_call(self, tool_call) -> None:
        """Fire-and-track ``handle_tool_call`` so multiple parallel tool calls
        in the same turn do not serialize behind one another and the Gemini
        receive loop stays responsive to cancellations and further events.
        """
        if self.tool_handler is None:
            return
        task = asyncio.create_task(self.tool_handler.handle_tool_call(tool_call))
        self._tool_dispatch_tasks.add(task)
        task.add_done_callback(self._on_tool_dispatch_done)

    def _on_tool_dispatch_done(self, task: asyncio.Task) -> None:
        self._tool_dispatch_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self.logger.error("[Orchestrator] Tool dispatch task crashed", error=str(exc))

    async def _handle_transport_messages(self):
        """Pipeline: transport -> Gemini."""
        async for message in self.transport.receive():
            match message:
                case AudioData() as audio:
                    self.metric_tracker.on_audio_sent()
                    if self.audio_recorder:
                        self.audio_recorder.record_user_audio(audio.data)
                    await self.gemini_session.send_audio(audio.data)
                case TextData() as text:
                    await self.gemini_session.send_text(text.text)
                case EventData() as event:
                    if self.callbacks.on_event:
                        await self.callbacks.on_event(event)
        self.logger.info("[Orchestrator] Transport loop ended")

    async def _handle_gemini_responses(self):
        """Pipeline: Gemini -> transport (+ tool handler).

        Thin dispatcher: every response type maps to a ``_on_*`` handler via
        ``self._response_handlers``. Unknown types are ignored.

        A single top-level ``RuntimeError`` catch handles the case where a
        ``transport.send_*`` call races the client's disconnect: the
        transport's ``_ensure_connected`` gate raises ``RuntimeError`` when
        the socket is no longer in ``CONNECTED`` state. We treat that as a
        normal end-of-session so ``_run_pipelines`` cancels the siblings and
        ``stop()`` runs cleanly, instead of crashing the task with a
        traceback.
        """
        try:
            async for response in self.gemini_session.receive():
                handler = self._response_handlers.get(response.type)
                if handler is not None:
                    await handler(response.data)
        except RuntimeError as e:
            self.logger.info("[Orchestrator] Transport closed mid-response; ending Gemini loop", error=str(e))
        self.logger.info("[Orchestrator] Gemini response loop ended")

    async def _on_audio(self, data: AudioData) -> None:
        self.metric_tracker.on_ttfb_end()
        self.metric_tracker.on_audio_received()
        if self.audio_recorder:
            self.audio_recorder.record_model_audio(data.data)
        await self.transport.write_audio(data)
        await self._model_vad.on_audio_chunk(data.data, data.sample_rate)

    async def _on_text(self, data: TextData) -> None:
        await self.transport.send_text(data)

    async def _on_transcript(self, data: TranscriptData) -> None:
        await self.transport.send_transcript(data)
        # Interim transcripts are provisional and may be revised by Gemini;
        # forward them for real-time display only, never into history/metrics.
        if data.interim:
            return
        if data.role == Role.USER:
            self.metric_tracker.on_user_transcript(data.text)
            if self.transcription:
                await self.transcription.on_user_transcript(data.text)
        elif data.role == Role.MODEL:
            self.metric_tracker.on_model_transcript(data.text)
            if self.transcription:
                await self.transcription.on_model_transcript(data.text)

    async def _on_turn_state_change(
        self,
        old: ConversationState,
        new: ConversationState,
    ) -> None:
        ttfb: Optional[float] = None
        if new == ConversationState.USER_TALKING:
            self.metric_tracker.on_user_turn()
        elif new == ConversationState.MODEL_TALKING:
            self.metric_tracker.on_model_turn()
            if self.metric_tracker._ttfb_samples:
                ttfb = self.metric_tracker._ttfb_samples[-1]

        if self.callbacks.on_turn_state_change:
            await self.callbacks.on_turn_state_change(old, new, ttfb)

    async def _on_interrupted(self, data: InterruptionData) -> None:
        self.metric_tracker.on_ttfb_cancel()
        self.metric_tracker.on_interruption()
        await self.transport.send_interruption(data)
        # force_stop emits a synthetic MODEL_END that the turn tracker will
        # handle; the upcoming USER_START VAD from Gemini will reset timers.
        await self._model_vad.force_stop()
        if self.transcription:
            await self.transcription.on_interrupted()
        if self.callbacks.on_interrupted:
            await self.callbacks.on_interrupted(data)

    async def _on_user_voice_activity(self, data: VoiceActivityData) -> None:
        self.logger.info(
            f"[Orchestrator] User voice activity state change: {data.voice_activity_type.value}",
            role=data.role.value,
            activity=data.voice_activity_type.value,
        )
        await self.transport.send_voice_activity(data)
        if data.voice_activity_type == types.VoiceActivityType.ACTIVITY_START:
            await self._turn_tracker.user_speech_start()
        elif data.voice_activity_type == types.VoiceActivityType.ACTIVITY_END:
            await self._turn_tracker.user_speech_end()
            self.metric_tracker.on_ttfb_start()
        if self.callbacks.on_voice_activity:
            await self.callbacks.on_voice_activity(data)

    async def _on_turn_complete(self, data: TurnCompleteData) -> None:
        # A previously frozen batch that never completed (rare, e.g. barge-in
        # mid-batch): flush only ITS buffered entries so they are not silently
        # dropped. Must filter by ID — _pending_batch may also hold results
        # buffered for the turn that is completing right now.
        if self._batch_tool_ids:
            stale = [
                self._pending_batch.pop(tool_id)
                for tool_id in list(self._pending_batch)
                if tool_id in self._batch_tool_ids
            ]
            self.logger.warning(
                "[Orchestrator] Stale tool response batch at turn boundary",
                stale_tool_ids=list(self._batch_tool_ids),
                flushed=len(stale),
            )
            self._batch_tool_ids.clear()
            if stale:
                try:
                    await self.gemini_session.send_tool_response_batch(stale)
                except Exception as e:
                    self.logger.error(
                        "[Orchestrator] Failed to flush stale tool responses",
                        error=str(e),
                    )

        # Freeze this turn's expected tool set — even for a single tool call —
        # so its responses go out together only after the turn has closed.
        if self._current_turn_tool_ids:
            self._batch_tool_ids = set(self._current_turn_tool_ids)
            self.logger.info(
                "[Orchestrator] Turn complete with tool calls; batching responses",
                tool_ids=self._current_turn_tool_ids,
                expected=len(self._batch_tool_ids),
                buffered=len(self._pending_batch),
            )
        self._current_turn_tool_ids = []

        # Fast tools may have finished before the turn closed — flush now.
        if self._batch_tool_ids and len(self._pending_batch) >= len(self._batch_tool_ids):
            await self._flush_batch()

        await self.transport.flush_audio()
        await self.transport.send_turn_complete(data)
        await self._model_vad.on_turn_complete()
        if self.transcription:
            await self.transcription.on_model_turn_complete()
        if self.callbacks.on_turn_complete:
            await self.callbacks.on_turn_complete(data)

    async def _on_tool_call(self, data: ToolCallData) -> None:
        self.metric_tracker.on_tool_call()
        if self.tool_handler:
            self._current_turn_tool_ids.append(data.id)
            self._dispatch_tool_call(data)
        else:
            self.logger.warning("[Orchestrator] Tool call received but no handler registered")

    async def _on_usage_metadata(self, data: UsageMetadataData) -> None:
        self.metric_tracker.on_usage_metadata(data)
        if self.callbacks.on_usage_metadata:
            await self.callbacks.on_usage_metadata(data)

    async def _on_tool_cancellation(self, data: ToolCallCancellationData) -> None:
        # Cancelled tools produce no FunctionResponse: drop them from every
        # stage of batch accounting (pre-freeze list, frozen set, buffered
        # results) so a batch can never wait on a cancelled function ID.
        for tool_id in data.ids:
            if tool_id in self._current_turn_tool_ids:
                self._current_turn_tool_ids.remove(tool_id)
            self._batch_tool_ids.discard(tool_id)
            self._pending_batch.pop(tool_id, None)

        if self._batch_tool_ids and len(self._pending_batch) >= len(self._batch_tool_ids):
            await self._flush_batch()

        if self.tool_handler:
            await self.tool_handler.handle_cancellation(data.ids)
        else:
            self.logger.warning("[Orchestrator] Tool cancellation received but no handler registered")

    async def _send_tool_result(self, result: ToolHandlerResult) -> None:
        """Forward a ``ToolHandlerResult`` to Gemini.

        Performs the final cancellation check immediately before the wire
        send, closing the race where a cancel arrives after the tool
        finishes executing but before its result reaches Gemini. The check
        must live here (not in the handler alone) because handler-side
        suppression happens before this ``await`` point — any cancel that
        lands on the event loop while we are waiting would otherwise leak
        a stale response as ghost context into the next turn.
        """
        if self.tool_handler and result.tool_id in self.tool_handler._cancelled_ids:
            self.tool_handler._cancelled_ids.pop(result.tool_id, None)
            self.logger.info(
                "[Orchestrator] Suppressed tool result send due to tool cancellation",
                action=result.action.value,
                tool_name=result.tool_name,
                tool_id=result.tool_id,
                reason="cancelled_before_send",
            )
            return

        # Hold every blocking response until its turn's TURN_COMPLETE so
        # Gemini always receives the turn's FunctionResponses as one batched
        # message. A response delivered mid-turn or as a partial set makes
        # Gemini re-emit tool calls it believes are unanswered.
        if result.action == ToolResponseAction.SEND_RESPONSE and (
            result.tool_id in self._batch_tool_ids
            or result.tool_id in self._current_turn_tool_ids
        ):
            self._pending_batch[result.tool_id] = result
            self.logger.info(
                "[Orchestrator] Buffered tool result until turn complete",
                tool_name=result.tool_name,
                tool_id=result.tool_id,
                buffered=len(self._pending_batch),
                expected=len(self._batch_tool_ids) or None,
                turn_open=result.tool_id in self._current_turn_tool_ids,
            )
            if self._batch_tool_ids and len(self._pending_batch) >= len(self._batch_tool_ids):
                await self._flush_batch()
            return

        # This result is answered outside the batch: a non-blocking
        # dispatch event (SEND_INTERIM — the function ID stays unanswered
        # until the scheduled FunctionResponse arrives), a non-blocking
        # final response, or a late response whose turn was already
        # settled (e.g. after a stale flush). Settle its batch accounting
        # so a frozen or future batch never waits on this ID.
        if result.tool_id in self._current_turn_tool_ids:
            self._current_turn_tool_ids.remove(result.tool_id)
        self._batch_tool_ids.discard(result.tool_id)

        try:
            match result.action:
                case ToolResponseAction.SEND_RESPONSE:
                    self.logger.info(
                        "[Orchestrator] Forwarding tool response to Gemini",
                        tool_name=result.tool_name,
                        tool_id=result.tool_id,
                        scheduling=result.scheduling.value if result.scheduling else None,
                    )
                    await self.gemini_session.send_tool_response(
                        result.tool_id, result.tool_name, result.result,
                        scheduling=result.scheduling,
                    )
                case ToolResponseAction.SEND_INTERIM:
                    # Doc-style acknowledgment: the interim message is sent
                    # verbatim as client text; the function call ID stays
                    # unanswered so the real result can be delivered later
                    # as a scheduled FunctionResponse.
                    if result.interim_message:
                        await self.gemini_session.send_text(result.interim_message)
        except Exception as e:
            self.logger.error(
                "[Orchestrator] Failed to forward tool result",
                error=str(e),
                tool_name=result.tool_name,
                tool_id=result.tool_id,
            )

        # A batch that was waiting only on this ID can now flush.
        if self._batch_tool_ids and len(self._pending_batch) >= len(self._batch_tool_ids):
            await self._flush_batch()

    async def _flush_batch(self) -> None:
        """Send all buffered blocking tool responses as a single batched FunctionResponse."""
        if not self._pending_batch:
            return

        batch = list(self._pending_batch.values())
        self._pending_batch.clear()
        self._batch_tool_ids.clear()

        self.logger.info(
            "[Orchestrator] Sending batched tool responses",
            batch_size=len(batch),
            tool_names=[r.tool_name for r in batch],
        )

        try:
            await self.gemini_session.send_tool_response_batch(batch)
        except Exception as e:
            self.logger.error(
                "[Orchestrator] Failed to send batched tool responses",
                error=str(e),
            )

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
        pending: set[asyncio.Task] = set()

        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                exc = task.exception() if not task.cancelled() else None
                if exc:
                    self.logger.error("[Orchestrator] Pipeline failed", error=str(exc))
        except Exception as e:
            self.logger.error(
                "[Orchestrator] Processing loop failed", error=str(e),
            )
            pending = {t for t in tasks if not t.done()}
        finally:
            if not pending:
                pending = {t for t in tasks if not t.done()}
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
        self.logger.info(
            f"[Orchestrator] Model voice activity state change: {data.voice_activity_type.value}",
            role=data.role.value,
            activity=data.voice_activity_type.value,
        )
        if data.voice_activity_type == types.VoiceActivityType.ACTIVITY_START:
            await self._turn_tracker.model_speech_start()
        elif data.voice_activity_type == types.VoiceActivityType.ACTIVITY_END:
            await self._turn_tracker.model_speech_end()
        await self.transport.send_voice_activity(data)
        if self.callbacks.on_voice_activity:
            await self.callbacks.on_voice_activity(data)
