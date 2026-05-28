"""Gemini Live API session — manages the WebSocket connection, audio streaming, and response parsing."""

import json
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Iterable, List, Optional, AsyncContextManager, Callable, Awaitable

from google import genai
from google.genai import types
from google.genai.live import AsyncSession
from config import settings

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


class GeminiLiveSession:
    """Manages a Gemini Live API session for real-time conversation."""

    def __init__(
        self,
        config: types.LiveConnectConfig,
        *,
        model: Optional[str] = None,
        initial_text: Optional[str] = None,
        on_connect: Optional[Callable[[AsyncSession], Awaitable[None]]] = None,
    ):
        self.config = config
        self.model = model or settings.GEMINI_LIVE_MODEL
        self.initial_text = initial_text
        self.on_connect = on_connect

        self.client: Optional[genai.Client] = None
        self.session: Optional[AsyncSession] = None
        self.session_context: Optional[AsyncContextManager[AsyncSession]] = None
        self.is_connected: bool = False
        self._audio_chunks_this_turn: int = 0
        self._session_logger: Optional[SessionLogger] = None

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

            self.logger.info("Connecting to Gemini Live API", model=self.model)

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

            self.logger.info("Connected successfully to Gemini Live API")

            if self.on_connect:
                await self.on_connect(self.session)

            if self.initial_text:
                await self.send_text(self.initial_text)

            return True

        except Exception as e:
            self.logger.error(
                "Connection to Gemini Live API failed", error=str(e)
            )
            self.is_connected = False
            return False

    async def receive(self) -> AsyncGenerator[GeminiLiveResponse, None]:
        """Yield responses from Gemini (audio, text, transcript, tool call, etc.).

        Thin driver: each SDK response is fanned out through the per-surface
        ``_parse_*`` helpers so the receive loop stays readable. The
        ``_audio_chunks_this_turn`` counter is carried on ``self`` because
        it is mutated by three different branches (AUDIO increments;
        INTERRUPTED / TURN_COMPLETE read+reset).
        """
        if not self.is_connected or not self.session:
            self.logger.warning("Not connected to Gemini, skipping receive")
            return

        try:
            self.logger.info("Receive loop started")
            self._audio_chunks_this_turn = 0

            while self.is_connected:
                async for response in self.session.receive():
                    for event in self._parse_response(response):
                        yield event
                    if not self.is_connected:
                        break

            self.logger.info("Receive loop ended")

        except Exception as e:
            self.logger.error("Receive loop failed", error=str(e))

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
            self.logger.info("Model playback interrupted", interrupted_chunks=interrupted_chunks)
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
            self.logger.info("Model turn complete", completed_chunks=completed_chunks)
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
                self.logger.error("Dropping FunctionCall with no id", tool_name=fc.name, args=args_dict)
                continue
            self.logger.info("Tool call received from Gemini", tool_name=fc.name, tool_call_id=fc.id, args=args_dict)
            yield GeminiLiveResponse(
                type=GeminiLiveResponseType.TOOL_CALL,
                data=ToolCallData(id=fc.id, name=fc.name or "", args=args_dict),
            )

    def _parse_tool_cancellation(self, cancellation) -> Iterable[GeminiLiveResponse]:
        cancelled_ids: list[str] = []
        if hasattr(cancellation, "ids") and cancellation.ids:
            cancelled_ids = list(cancellation.ids)
        self.logger.info("Tool call cancellation received", cancelled_ids=cancelled_ids)
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
            "Usage metadata received",
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
        """Send PCM16 16 kHz audio to Gemini."""
        if not self.is_connected or not self.session:
            self.logger.debug("Not connected to Gemini Live, skipping send_audio")
            return

        try:
            await self.session.send_realtime_input(
                audio=types.Blob(data=audio_data, mime_type="audio/pcm;rate=16000")
            )
        except Exception as e:
            self.logger.error("Send audio failed", error=str(e))

    async def send_tool_response(
        self, function_id: str, function_name: str, response: Any
    ) -> None:
        """Send a FunctionResponse to Gemini (blocking tools).

        Gemini is waiting for a FunctionResponse matching the tool call ID,
        so this does NOT trigger an interruption.
        """
        if not self.session or not self.is_connected:
            self.logger.error("Not connected, skipping send_tool_response", tool_name=function_name, tool_call_id=function_id)
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
        self.logger.info("FunctionResponse sent back to Gemini", tool_name=function_name, tool_call_id=function_id)

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
            self.logger.error("Not connected, skipping send_tool_result_as_context", tool_name=function_name, tool_call_id=function_id)
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
        self.logger.info("Context result sent back to Gemini", tool_name=function_name, tool_call_id=function_id)

    async def send_interim_tool_response(
        self, function_id: str, function_name: str, interim_message: str
    ) -> None:
        """Send an interim PROCESSING FunctionResponse to unblock the model
        for speech while the tool executes in the background.
        """
        if not self.session or not self.is_connected:
            self.logger.error("Not connected, skipping send_interim_tool_response", tool_name=function_name, tool_call_id=function_id)
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
            self.logger.info("Interim processing response sent to Gemini", tool_name=function_name, tool_call_id=function_id)
        except Exception as e:
            self.logger.error(
                "Interim response failed to send",
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
            self.logger.debug("Not connected, skipping send_text")
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
            self.logger.error("Send text failed", error=str(e))

    async def send_system_text(self, text: str) -> None:
        """Send system text input to Gemini."""
        if not self.is_connected or not self.session:
            self.logger.error("Not connected, skipping send_system_text")
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
            self.logger.error("Send system text failed", error=str(e))

    async def disconnect(self):
        """Disconnect from Gemini Live API and clean up resources."""
        try:
            if self.session_context:
                try:
                    await self.session_context.__aexit__(None, None, None)
                    self.logger.info("Disconnected from Gemini Live API")
                except Exception as e:
                    self.logger.error("Disconnect failed", error=str(e))
        finally:
            self.session = None
            self.session_context = None
            self.is_connected = False

