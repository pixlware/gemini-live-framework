"""Gemini Live API session — manages the WebSocket connection, audio streaming, and response parsing."""

import asyncio
import json
from enum import Enum
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Iterable, List, Optional, AsyncContextManager, Callable, Awaitable

from google import genai
from google.genai import types
from google.genai.live import AsyncSession
from config import settings

from .silero_vad import SileroVad

from .models import (
    AudioFormat,
    AudioData,
    TextData,
    TranscriptData,
    Role,
    ToolCallData,
    ToolCallCancellationData,
    VoiceActivityData,
    UsageMetadataData,
    Data,
    InterruptionData,
    TurnCompleteData,
)

from .logger import SessionLogger

class GeminiLiveResponseType(Enum):
    AUDIO = "audio"
    TEXT = "text"
    TRANSCRIPT = "transcript"
    TOOL_CALL = "tool_call"
    TOOL_CALL_CANCELLATION = "tool_call_cancellation"
    TURN_COMPLETE = "turn_complete"
    INTERRUPTED = "interrupted"
    VOICE_ACTIVITY = "voice_activity"
    USAGE_METADATA = "usage_metadata"


@dataclass
class GeminiLiveResponse:
    type: GeminiLiveResponseType
    data: Optional[Data] = None


# Marker pushed onto the response queue when the background receive task ends,
# so the public receive() consumer can exit cleanly instead of hanging on get().
_SENTINEL = object()


class GeminiLiveSession:
    """Manages a Gemini Live API session for real-time conversation."""

    def __init__(
        self,
        config: types.LiveConnectConfig,
        *,
        model: Optional[str] = None,
        initial_text: Optional[str] = None,
        on_connect: Optional[Callable[[AsyncSession], Awaitable[None]]] = None,
        vad_type: str = "gemini",
    ):
        self.config = config
        self.model = model or settings.GEMINI_LIVE_MODEL
        self.initial_text = initial_text
        self.on_connect = on_connect
        self.vad_type = vad_type

        self.client: Optional[genai.Client] = None
        self.session: Optional[AsyncSession] = None
        self.session_context: Optional[AsyncContextManager[AsyncSession]] = None
        self.is_connected: bool = False
        self._audio_chunks_this_turn: int = 0
        self._session_logger: Optional[SessionLogger] = None

        # Client-side VAD (eager, no lazy load) — only when explicitly selected.
        self._silero_vad: Optional[SileroVad] = (
            SileroVad() if vad_type == "silero" else None
        )

        # Unbounded so put_nowait never blocks or raises QueueFull. Both the
        # background network drain and the Silero send path feed this queue;
        # the public receive() generator is the sole consumer.
        self._responses: asyncio.Queue = asyncio.Queue()
        self._receive_responses_task: Optional[asyncio.Task] = None

    @property
    def logger(self) -> SessionLogger:
        if self._session_logger is None:
            self._session_logger = SessionLogger()
        return self._session_logger

    @logger.setter
    def logger(self, logger: SessionLogger) -> None:
        self._session_logger = logger

    async def connect(self) -> bool:
        """Establish connection to Gemini Live API.  Returns True on success."""
        try:
            self.client = genai.Client(
                vertexai=True,
                project=settings.GOOGLE_CLOUD_PROJECT,
                location=settings.GEMINI_LOCATION,
            )

            self.logger.info("[GeminiSession] Connecting to Gemini Live API", model=self.model)

            session_context = self.client.aio.live.connect(
                model=self.model,
                config=self.config,
            )

            self.session = await session_context.__aenter__()
            self.session_context = session_context
            self.is_connected = True

            session_id = self.session.session_id if self.session else None

            # Auto-bind gemini_session_id upon connection to propagate across all session log statements
            if session_id:
                self.logger.bind(gemini_session_id=session_id)

            self.logger.info("[GeminiSession] Connected successfully to Gemini Live API")

            # Fresh queue per connection so a reconnect never inherits stale
            # events or a leftover sentinel. Start the background drain before
            # on_connect/initial_text so responses to the initial turn are
            # captured.
            self._responses = asyncio.Queue()
            self._receive_responses_task = asyncio.create_task(
                self._receive_responses()
            )

            if self.on_connect:
                await self.on_connect(self.session)

            if self.initial_text:
                await self.send_text(self.initial_text)

            return True

        except Exception as e:
            self.logger.error(
                "[GeminiSession] Connection to Gemini Live API failed", error=str(e)
            )
            self.is_connected = False
            return False

    async def _receive_responses(self) -> None:
        """Background producer: drain the Gemini network stream into the queue.

        Each SDK response is fanned out through the per-surface ``_parse_*``
        helpers and pushed onto ``self._responses`` via ``put_nowait``. The
        ``_audio_chunks_this_turn`` counter is carried on ``self`` because it
        is mutated by three different branches (AUDIO increments; INTERRUPTED /
        TURN_COMPLETE read+reset). On exit, a sentinel is pushed so the public
        ``receive()`` consumer can terminate cleanly.
        """
        if self.session is None:
            self._responses.put_nowait(_SENTINEL)
            return

        self.logger.info("[GeminiSession] Receive loop started")
        self._audio_chunks_this_turn = 0

        try:
            while self.is_connected:
                async for response in self.session.receive():
                    for event in self._parse_response(response):
                        self._responses.put_nowait(event)
                    if not self.is_connected:
                        break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.error("[GeminiSession] Receive loop failed", error=str(e))
        finally:
            self._responses.put_nowait(_SENTINEL)
            self.logger.info("[GeminiSession] Receive loop ended")

    async def receive(self) -> AsyncGenerator[GeminiLiveResponse, None]:
        """Yield responses from Gemini (audio, text, transcript, tool call, etc.).

        Consumer side: drains the unified ``self._responses`` queue fed by the
        background ``_receive_responses`` task (Gemini network) and by the
        Silero send path. Exits when the sentinel is observed.
        """
        while True:
            event = await self._responses.get()
            if event is _SENTINEL:
                break
            yield event

    def _parse_response(self, response: types.LiveServerMessage) -> Iterable[GeminiLiveResponse]:
        """Fan out one SDK response into zero or more ``GeminiLiveResponse`` events."""
        if response.data is not None:
            yield from self._parse_audio(response.data)
        if response.server_content:
            yield from self._parse_server_content(response.server_content)
        if response.tool_call and response.tool_call.function_calls:
            yield from self._parse_tool_calls(response.tool_call.function_calls)
        if response.tool_call_cancellation:
            yield from self._parse_tool_cancellation(response.tool_call_cancellation)
        if response.voice_activity:
            yield from self._parse_voice_activity(response.voice_activity)
        if response.usage_metadata:
            yield from self._parse_usage_metadata(response.usage_metadata)

    def _parse_audio(self, raw_audio: bytes) -> Iterable[GeminiLiveResponse]:
        self._audio_chunks_this_turn += 1
        yield GeminiLiveResponse(
            type=GeminiLiveResponseType.AUDIO,
            data=AudioData(
                data=raw_audio,
                format=AudioFormat.PCM16,
                sample_rate=24000,
            ),
        )

    def _parse_server_content(self, sc: types.LiveServerContent) -> Iterable[GeminiLiveResponse]:
        if sc.interrupted:
            interrupted_chunks = self._audio_chunks_this_turn
            self.logger.info("[GeminiSession] Model playback interrupted", interrupted_chunks=interrupted_chunks)
            self._audio_chunks_this_turn = 0
            yield GeminiLiveResponse(
                type=GeminiLiveResponseType.INTERRUPTED,
                data=InterruptionData(audio_chunks=interrupted_chunks),
            )

        if sc.model_turn:
            for part in sc.model_turn.parts or []:
                if part.text:
                    yield GeminiLiveResponse(
                        type=GeminiLiveResponseType.TEXT,
                        data=TextData(text=part.text),
                    )

        if sc.input_transcription and sc.input_transcription.text:
            yield GeminiLiveResponse(
                type=GeminiLiveResponseType.TRANSCRIPT,
                data=TranscriptData(role=Role.USER, text=sc.input_transcription.text),
            )

        if sc.output_transcription and sc.output_transcription.text:
            yield GeminiLiveResponse(
                type=GeminiLiveResponseType.TRANSCRIPT,
                data=TranscriptData(role=Role.MODEL, text=sc.output_transcription.text),
            )

        if sc.turn_complete:
            completed_chunks = self._audio_chunks_this_turn
            self.logger.info("[GeminiSession] Model turn complete", completed_chunks=completed_chunks)
            self._audio_chunks_this_turn = 0
            yield GeminiLiveResponse(
                type=GeminiLiveResponseType.TURN_COMPLETE,
                data=TurnCompleteData(audio_chunks=completed_chunks),
            )

    def _parse_tool_calls(self, function_calls: List[types.FunctionCall]) -> Iterable[GeminiLiveResponse]:
        for fc in function_calls:
            try:
                args_dict = dict(fc.args) if fc.args else {}
            except (TypeError, AttributeError):
                args_dict = {}
            if not fc.id:
                self.logger.error("[GeminiSession] Dropping FunctionCall with no id", tool_name=fc.name, args=args_dict)
                continue
            self.logger.info("[GeminiSession] Tool call received from Gemini", tool_name=fc.name, tool_call_id=fc.id, args=args_dict)
            yield GeminiLiveResponse(
                type=GeminiLiveResponseType.TOOL_CALL,
                data=ToolCallData(id=fc.id, name=fc.name or "", args=args_dict),
            )

    def _parse_tool_cancellation(self, cancellation) -> Iterable[GeminiLiveResponse]:
        cancelled_ids: list[str] = []
        if hasattr(cancellation, "ids") and cancellation.ids:
            cancelled_ids = list(cancellation.ids)
        self.logger.info("[GeminiSession] Tool call cancellation received", cancelled_ids=cancelled_ids)
        yield GeminiLiveResponse(
            type=GeminiLiveResponseType.TOOL_CALL_CANCELLATION,
            data=ToolCallCancellationData(ids=cancelled_ids),
        )

    def _parse_voice_activity(self, va: types.VoiceActivity) -> Iterable[GeminiLiveResponse]:
        if not va.voice_activity_type:
            return
        if va.voice_activity_type == types.VoiceActivityType.TYPE_UNSPECIFIED:
            return
        yield GeminiLiveResponse(
            type=GeminiLiveResponseType.VOICE_ACTIVITY,
            data=VoiceActivityData(
                role=Role.USER,
                voice_activity_type=va.voice_activity_type,
            ),
        )

    def _parse_usage_metadata(self, usage: types.UsageMetadata) -> Iterable[GeminiLiveResponse]:
        data = UsageMetadataData(
            prompt_token_count=usage.prompt_token_count or 0,
            response_token_count=usage.response_token_count or 0,
            total_token_count=usage.total_token_count or 0,
            thoughts_token_count=usage.thoughts_token_count or 0,
            tool_use_prompt_token_count=usage.tool_use_prompt_token_count or 0,
        )
        self.logger.info(
            "[GeminiSession] Usage metadata received",
            prompt_tokens=data.prompt_token_count,
            response_tokens=data.response_token_count,
            total_tokens=data.total_token_count,
            thoughts_tokens=data.thoughts_token_count,
            tool_use_tokens=data.tool_use_prompt_token_count,
        )
        yield GeminiLiveResponse(
            type=GeminiLiveResponseType.USAGE_METADATA,
            data=data,
        )

    async def send_audio(self, audio_data: bytes) -> None:
        """Send PCM16 16 kHz audio to Gemini.

        In ``gemini`` mode the audio streams continuously and the server runs
        VAD. In ``silero`` mode the audio is gated by the local detector and
        manual activity signals are sent instead.
        """
        if not self.is_connected or not self.session:
            return

        try:
            if self.vad_type == "silero" and self._silero_vad is not None:
                await self._send_audio_silero(audio_data)
            else:
                await self.session.send_realtime_input(
                    audio=types.Blob(data=audio_data, mime_type="audio/pcm;rate=16000")
                )
        except Exception as e:
            self.logger.error("[GeminiSession] Send audio failed", error=str(e))

    async def _send_audio_silero(self, audio_data: bytes) -> None:
        """Drive the local Silero detector and forward gated audio + signals.

        For each VAD transition the local ``VoiceActivityData`` event is queued
        first (instant, no network wait) so the orchestrator reacts to speech
        start/end without waiting on Gemini. ``send_realtime_input`` accepts
        exactly one argument per call, so the manual signal and the audio blob
        are sent as separate awaits.
        """
        vad = self._silero_vad
        if self.session is None or vad is None:
            return

        for signal, audio in vad.process_audio(audio_data):
            if signal == "start":
                self._responses.put_nowait(GeminiLiveResponse(
                    type=GeminiLiveResponseType.VOICE_ACTIVITY,
                    data=VoiceActivityData(
                        role=Role.USER,
                        voice_activity_type=types.VoiceActivityType.ACTIVITY_START,
                    ),
                ))
                await self.session.send_realtime_input(activity_start=types.ActivityStart())
                if audio:
                    await self.session.send_realtime_input(
                        audio=types.Blob(data=audio, mime_type="audio/pcm;rate=16000")
                    )
            elif signal == "end":
                self._responses.put_nowait(GeminiLiveResponse(
                    type=GeminiLiveResponseType.VOICE_ACTIVITY,
                    data=VoiceActivityData(
                        role=Role.USER,
                        voice_activity_type=types.VoiceActivityType.ACTIVITY_END,
                    ),
                ))
                await self.session.send_realtime_input(activity_end=types.ActivityEnd())
            else:
                if audio:
                    await self.session.send_realtime_input(
                        audio=types.Blob(data=audio, mime_type="audio/pcm;rate=16000")
                    )

    async def send_tool_response(
        self, function_id: str, function_name: str, response: Any
    ) -> None:
        """Send a FunctionResponse to Gemini (blocking tools).

        Gemini is waiting for a FunctionResponse matching the tool call ID,
        so this does NOT trigger an interruption.
        """
        if not self.session or not self.is_connected:
            self.logger.error("[GeminiSession] Not connected, skipping send_tool_response", tool_name=function_name, tool_call_id=function_id)
            return

        result_payload = response if isinstance(response, dict) else {"result": response}
        func_response = types.FunctionResponse(
            id=function_id,
            name=function_name,
            response=result_payload,
        )
        await self.session.send_tool_response(
            function_responses=[func_response]
        )
        self.logger.info("[GeminiSession] FunctionResponse sent back to Gemini", tool_name=function_name, tool_call_id=function_id)

    async def send_tool_result_as_context(
        self, function_id: str, function_name: str, response: Any
    ) -> None:
        """Inject a tool result as client content (non-blocking tools only).

        The interim PROCESSING FunctionResponse already consumed the
        function_id, so a second FunctionResponse with the same ID would
        make the model repeat itself.  Instead we inject the result as
        client content so the model speaks about it naturally.
        """
        if not self.session or not self.is_connected:
            self.logger.error("[GeminiSession] Not connected, skipping send_tool_result_as_context", tool_name=function_name, tool_call_id=function_id)
            return

        result_text = (
            f"[Tool completed] {function_name} result: "
            f"{json.dumps(response, default=str)}"
        )
        await self.session.send_client_content(
            turns=types.Content(
                parts=[types.Part(text=result_text)],
                role="user",
            ),
            turn_complete=True,
        )
        self.logger.info("[GeminiSession] Context result sent back to Gemini", tool_name=function_name, tool_call_id=function_id)

    async def send_interim_tool_response(
        self, function_id: str, function_name: str, interim_message: str
    ) -> None:
        """Send an interim PROCESSING FunctionResponse to unblock the model
        for speech while the tool executes in the background.
        """
        if not self.session or not self.is_connected:
            self.logger.error("[GeminiSession] Not connected, skipping send_interim_tool_response", tool_name=function_name, tool_call_id=function_id)
            return

        try:
            interim = types.FunctionResponse(
                id=function_id,
                name=function_name,
                response={"status": "PROCESSING", "message": interim_message},
            )
            await self.session.send_tool_response(
                function_responses=[interim]
            )
            self.logger.info("[GeminiSession] Interim processing response sent to Gemini", tool_name=function_name, tool_call_id=function_id)
        except Exception as e:
            self.logger.error(
                "[GeminiSession] Interim response failed to send",
                tool_name=function_name,
                tool_call_id=function_id,
                error=str(e),
            )

    async def send_text(self, text: str, turn_complete: bool = True) -> None:
        """Send text input to Gemini.

        *turn_complete* must be True when used alongside send_realtime_input
        to prevent the text turn from colliding with the audio stream.
        """
        if not self.is_connected or not self.session:
            self.logger.debug("[GeminiSession] Not connected, skipping send_text")
            return

        try:
            await self.session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part(text=text)]
                ),
                turn_complete=turn_complete,
            )
        except Exception as e:
            self.logger.error("[GeminiSession] Send text failed", error=str(e))

    async def send_system_text(self, text: str) -> None:
        """Send system text input to Gemini."""
        if not self.is_connected or not self.session:
            self.logger.error("[GeminiSession] Not connected, skipping send_system_text")
            return
        try:
            await self.session.send_client_content(
                turns=types.Content(
                    role="system",
                    parts=[types.Part(text=text)]
                ),
                turn_complete=False,
            )
        except Exception as e:
            self.logger.error("[GeminiSession] Send system text failed", error=str(e))

    async def disconnect(self):
        """Disconnect from Gemini Live API and clean up resources."""
        self.is_connected = False

        # Stop the background drain before tearing down the session so it
        # cannot outlive the connection it reads from.
        if self._receive_responses_task:
            self._receive_responses_task.cancel()
            try:
                await self._receive_responses_task
            except asyncio.CancelledError:
                pass
            self._receive_responses_task = None

        try:
            if self.session_context:
                try:
                    await self.session_context.__aexit__(None, None, None)
                    self.logger.info("[GeminiSession] Disconnected from Gemini Live API")
                except Exception as e:
                    self.logger.error("[GeminiSession] Disconnect failed", error=str(e))
        finally:
            self.session = None
            self.session_context = None
            self.is_connected = False

