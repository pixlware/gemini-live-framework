"""Gemini Live API session — manages the WebSocket connection, audio streaming, and response parsing."""

import asyncio
import json
import logging
from enum import Enum
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional, AsyncContextManager, Callable, Awaitable

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
    TurnCompleteData,
)

logger = logging.getLogger(__name__)


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

    model: str = "gemini-live-2.5-flash-native-audio"
    client: Optional[genai.Client] = None
    session: Optional[AsyncSession] = None
    session_context: Optional[AsyncContextManager[AsyncSession]] = None
    is_connected: bool = False

    system_instruction: Optional[str] = None
    function_declarations: Optional[list[types.FunctionDeclaration]] = None
    rag_corpus: Optional[str] = None

    voice_name: str = "Zephyr"
    language_code: Optional[str] = None

    input_audio_transcription: Optional[types.AudioTranscriptionConfig] = None
    output_audio_transcription: Optional[types.AudioTranscriptionConfig] = None

    vad_enabled: bool = True

    initial_text: Optional[str] = None

    on_connect: Optional[Callable[[AsyncSession], Awaitable[None]]] = None

    def __init__(
        self,
        voice_name: str = "Zephyr",
        language_code: Optional[str] = None,
        system_instruction: Optional[str] = None,
        function_declarations: Optional[list[types.FunctionDeclaration]] = None,
        rag_corpus: Optional[str] = None,
        input_audio_transcription: Optional[types.AudioTranscriptionConfig] = None,
        output_audio_transcription: Optional[types.AudioTranscriptionConfig] = None,
        vad_enabled: Optional[bool] = True,
        initial_text: Optional[str] = None,
        on_connect: Optional[Callable[[AsyncSession], Awaitable[None]]] = None,
    ):
        self.system_instruction = system_instruction
        self.function_declarations = function_declarations
        self.rag_corpus = rag_corpus
        self.voice_name = voice_name
        self.language_code = language_code
        self.input_audio_transcription = input_audio_transcription
        self.output_audio_transcription = output_audio_transcription
        self.vad_enabled = vad_enabled
        self.initial_text = initial_text
        self.on_connect = on_connect

    async def connect(self) -> bool:
        """Establish connection to Gemini Live API.  Returns True on success."""
        try:
            self.client = genai.Client(
                vertexai=True,
                project=settings.GOOGLE_CLOUD_PROJECT,
                location="us-central1"
            )

            logger.info("[GeminiLiveSession] Connecting")

            system_instruction = self._get_system_instruction()
            tools_config = self._get_tool_config()
            vad_config = types.AutomaticActivityDetection(
                disabled=True
            )
            if self.vad_enabled:
                vad_config = types.AutomaticActivityDetection(
                    disabled=False,
                    start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_UNSPECIFIED,
                    end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                    prefix_padding_ms=300,
                    silence_duration_ms=800,
                )

            input_audio_transcription = self.input_audio_transcription
            if not input_audio_transcription:
                input_audio_transcription = types.AudioTranscriptionConfig(
                    language_codes=['en-US']
                )
            output_audio_transcription = self.output_audio_transcription
            if not output_audio_transcription:
                output_audio_transcription = types.AudioTranscriptionConfig(
                    language_codes=['en-US']
                )

            config = types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self.voice_name
                        )
                    ),
                    language_code=self.language_code
                ),
                system_instruction=system_instruction,
                tools=tools_config or None,
                input_audio_transcription=input_audio_transcription,
                output_audio_transcription=output_audio_transcription,
                explicit_vad_signal=self.vad_enabled,
                realtime_input_config=types.RealtimeInputConfig(
                    automatic_activity_detection=vad_config
                ),
            )

            session_context = self.client.aio.live.connect(
                model=self.model,
                config=config
            )

            self.session = await session_context.__aenter__()
            self.session_context = session_context
            self.is_connected = True

            session_id = self.session.session_id if self.session else None
            logger.info("[GeminiLiveSession] Connected. Session ID: %s", session_id)

            if self.on_connect:
                await self.on_connect(self.session)

            if self.initial_text:
                await self.send_text(self.initial_text)

            return True

        except Exception as e:
            logger.error(f"[GeminiLiveSession] Connection failed: {e}")
            self.is_connected = False
            return False

    async def receive_responses(self) -> AsyncGenerator[GeminiLiveResponse, None]:
        """Yield responses from Gemini (audio, text, transcript, tool call, etc.)."""
        if not self.is_connected or not self.session:
            logger.warning("[GeminiLiveSession] Not connected, skipping receive")
            return

        try:
            logger.info("[GeminiLiveSession] Receive loop started")
            _audio_chunk_count = 0

            while self.is_connected:
                had_activity = False

                async for response in self.session.receive():
                    had_activity = True

                    if response.data is not None:
                        _audio_chunk_count += 1
                        audio_data = AudioData(
                            data=response.data,
                            format=AudioFormat.PCM16,
                            sample_rate=24000
                        )
                        yield GeminiLiveResponse(
                            type=GeminiLiveResponseType.AUDIO,
                            data=audio_data,
                        )

                    if response.server_content:
                        if response.server_content.interrupted:
                            logger.info(f"[GeminiLiveSession] Interrupted chunks={_audio_chunk_count}")
                            _audio_chunk_count = 0
                            yield GeminiLiveResponse(type=GeminiLiveResponseType.INTERRUPTED)

                        if response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                if part.text:
                                    text_data = TextData(text=part.text)
                                    yield GeminiLiveResponse(
                                        type=GeminiLiveResponseType.TEXT,
                                        data=text_data,
                                    )

                        if response.server_content.input_transcription:
                            if response.server_content.input_transcription.text:
                                user_text = response.server_content.input_transcription.text
                                transcript_data = TranscriptData(role=Role.USER, text=user_text)
                                yield GeminiLiveResponse(
                                    type=GeminiLiveResponseType.TRANSCRIPT,
                                    data=transcript_data,
                                )

                        if response.server_content.output_transcription:
                            if response.server_content.output_transcription.text:
                                model_text = response.server_content.output_transcription.text
                                transcript_data = TranscriptData(role=Role.MODEL, text=model_text)
                                yield GeminiLiveResponse(
                                    type=GeminiLiveResponseType.TRANSCRIPT,
                                    data=transcript_data,
                                )

                        if response.server_content.turn_complete:
                            completed_chunks = _audio_chunk_count
                            logger.info(
                                f"[GeminiLiveSession] Model turn complete. Audio chunks={completed_chunks}"
                            )
                            _audio_chunk_count = 0
                            yield GeminiLiveResponse(
                                type=GeminiLiveResponseType.TURN_COMPLETE,
                                data=TurnCompleteData(audio_chunks=completed_chunks),
                            )

                    if response.tool_call and response.tool_call.function_calls:
                        for fc in response.tool_call.function_calls:
                            try:
                                args_dict = dict(fc.args) if fc.args else {}
                            except (TypeError, AttributeError):
                                args_dict = {}
                            tool_call_id = fc.id if fc.id else f"tc_{uuid.uuid4().hex[:12]}"
                            if not fc.id:
                                logger.warning(f"[GeminiLiveSession] Tool call missing ID, generated tool={fc.name} id={tool_call_id}")
                            logger.info(f"[GeminiLiveSession] Tool call received tool={fc.name} id={tool_call_id} args={args_dict}")
                            yield GeminiLiveResponse(
                                type=GeminiLiveResponseType.TOOL_CALL,
                                data=ToolCallData(
                                    id=tool_call_id,
                                    name=fc.name,
                                    args=args_dict,
                                ),
                            )

                    if response.tool_call_cancellation:
                        cancelled_ids = []
                        if hasattr(response.tool_call_cancellation, 'ids') and response.tool_call_cancellation.ids:
                            cancelled_ids = list(response.tool_call_cancellation.ids)
                        logger.info(f"[GeminiLiveSession] Tool call cancellation ids={cancelled_ids}")
                        yield GeminiLiveResponse(
                            type=GeminiLiveResponseType.TOOL_CALL_CANCELLATION,
                            data=ToolCallCancellationData(ids=cancelled_ids),
                        )

                    if response.voice_activity:
                        if response.voice_activity.voice_activity_type:
                            voice_activity_type = response.voice_activity.voice_activity_type
                            if voice_activity_type != types.VoiceActivityType.TYPE_UNSPECIFIED:
                                yield GeminiLiveResponse(
                                    type=GeminiLiveResponseType.VOICE_ACTIVITY,
                                    data=VoiceActivityData(
                                        role=Role.USER,
                                        voice_activity_type=voice_activity_type
                                    ),
                                )

                    if response.usage_metadata:
                        usage_metadata = response.usage_metadata
                        usage_metadata_data = UsageMetadataData(
                            prompt_token_count=getattr(usage_metadata, 'prompt_token_count', 0) or 0,
                            response_token_count=getattr(usage_metadata, 'response_token_count', 0) or 0,
                            total_token_count=getattr(usage_metadata, 'total_token_count', 0) or 0,
                            thoughts_token_count=getattr(usage_metadata, 'thoughts_token_count', 0) or 0,
                            tool_use_prompt_token_count=getattr(usage_metadata, 'tool_use_prompt_token_count', 0) or 0,
                        )
                        logger.info(
                            f"[GeminiLiveSession] Usage metadata: "
                            f"prompt_token_count={usage_metadata_data.prompt_token_count}, "
                            f"response_token_count={usage_metadata_data.response_token_count}, "
                            f"total_token_count={usage_metadata_data.total_token_count}, "
                            f"thoughts_token_count={usage_metadata_data.thoughts_token_count}, "
                            f"tool_use_prompt_token_count={usage_metadata_data.tool_use_prompt_token_count}"
                        )
                        yield GeminiLiveResponse(
                            type=GeminiLiveResponseType.USAGE_METADATA,
                            data=usage_metadata_data,
                        )

                    if not self.is_connected:
                        break

                # CRITICAL: Sleep to prevent WebSocket timeout when no activity
                if not had_activity and self.is_connected:
                    await asyncio.sleep(0.1)

            logger.info("[GeminiLiveSession] Receive loop ended")

        except Exception as e:
            logger.error(f"[GeminiLiveSession] Receive loop failed: {e}", exc_info=True)

    async def send_audio(self, audio_data: bytes) -> bool:
        """Send PCM16 16 kHz audio to Gemini.  Returns True on success."""
        if not self.is_connected or not self.session:
            logger.debug("[GeminiLiveSession] Not connected, skipping send_audio")
            return False

        try:
            await self.session.send_realtime_input(
                audio=types.Blob(data=audio_data, mime_type="audio/pcm;rate=16000")
            )
            return True

        except Exception as e:
            logger.error(f"[GeminiLiveSession] Send audio failed: {e}")
            return False

    async def send_tool_response(
        self, function_id: str, function_name: str, response: Any
    ) -> None:
        """Send a FunctionResponse to Gemini (blocking tools).

        Gemini is waiting for a FunctionResponse matching the tool call ID,
        so this does NOT trigger an interruption.
        """
        if not self.session or not self.is_connected:
            logger.error("[GeminiLiveSession] Not connected, skipping send_tool_response")
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
        logger.info(f"[GeminiLiveSession] FunctionResponse sent tool={function_name} id={function_id}")

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
            logger.error("[GeminiLiveSession] Not connected, skipping send_tool_result_as_context")
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
        logger.info(f"[GeminiLiveSession] Context result sent tool={function_name} id={function_id}")

    async def send_interim_tool_response(
        self, function_id: str, function_name: str, interim_message: str
    ) -> None:
        """Send an interim PROCESSING FunctionResponse to unblock the model
        for speech while the tool executes in the background.
        """
        if not self.session or not self.is_connected:
            logger.error("[GeminiLiveSession] Not connected, skipping send_interim_tool_response")
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
            logger.info(f"[GeminiLiveSession] Interim response sent tool={function_name} id={function_id}")
        except Exception as e:
            logger.error(f"[GeminiLiveSession] Interim response failed tool={function_name}: {e}")

    async def signal_user_turn_complete(self) -> bool:
        """Signal that the user's turn is complete so the model responds.

        Call after sending the final silence burst for VAD-ended turns.
        """
        if not self.is_connected or not self.session:
            return False
        try:
            await self.session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part(text=" ")],
                ),
                turn_complete=True,
            )
            logger.debug("[GeminiLiveSession] User turn complete signaled")
            return True
        except Exception as e:
            logger.error(f"[GeminiLiveSession] Signal turn complete failed: {e}")
            return False

    async def send_text(self, text: str, turn_complete: bool = True) -> bool:
        """Send text input to Gemini.

        *turn_complete* must be True when used alongside send_realtime_input
        to prevent the text turn from colliding with the audio stream.
        """
        if not self.is_connected or not self.session:
            return False

        try:
            await self.session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part(text=text)]
                ),
                turn_complete=turn_complete,
            )
            return True

        except Exception as e:
            logger.error(f"[GeminiLiveSession] Send text failed: {e}")
            return False

    async def send_system_text(self, text: str) -> bool:
        """Send system text input to Gemini."""
        if not self.is_connected or not self.session:
            return False
        try:
            await self.session.send_client_content(
                turns=types.Content(
                    role="system",
                    parts=[types.Part(text=text)]
                ),
                turn_complete=False,
            )
            return True
        except Exception as e:
            logger.error(f"[GeminiLiveSession] Send system text failed: {e}")
            return False

    async def disconnect(self):
        """Disconnect from Gemini Live API and clean up resources."""
        if self.session_context:
            try:
                await self.session_context.__aexit__(None, None, None)
                logger.info("[GeminiLiveSession] Disconnected")
            except Exception as e:
                logger.error(f"[GeminiLiveSession] Disconnect failed: {e}")
            finally:
                self.session = None
                self.session_context = None
                self.is_connected = False

    def _get_tool_config(self) -> Optional[types.ToolListUnion]:
        tools_config = []

        if self.rag_corpus:
            rag_store = types.VertexRagStore(
                rag_resources=[
                    types.VertexRagStoreRagResource(
                        rag_corpus=self.rag_corpus
                    )
                ]
            )

            rag_tool = types.Tool(
                retrieval=types.Retrieval(vertex_rag_store=rag_store)
            )
            tools_config.append(rag_tool)

        if self.function_declarations:
            tools_config.append(types.Tool(function_declarations=self.function_declarations))

        return tools_config if tools_config and len(tools_config) > 0 else None

    def _get_system_instruction(self) -> Optional[str]:
        return self.system_instruction
