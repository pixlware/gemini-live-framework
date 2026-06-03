from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional

from fastapi import WebSocket
from pydantic import BaseModel, ConfigDict, PrivateAttr
from starlette.types import Message
from starlette.websockets import WebSocketDisconnect, WebSocketState

from ..audio_filters.base_audio_filter import BaseAudioFilter
from ..audio_transcoder import AudioTranscoder, build_transcoder
from ..buffer_service import BufferService
from ..models import (
    AudioData,
    AudioFormat,
    Data,
    EventData,
    TextData,
    InterruptionData,
    TranscriptData,
    VoiceActivityData,
    TurnCompleteData,
)

from ..logger import SessionLogger

FRAMEWORK_INPUT_FORMAT = AudioFormat.PCM16
FRAMEWORK_INPUT_SAMPLE_RATE = 16000
FRAMEWORK_OUTPUT_FORMAT = AudioFormat.PCM16
FRAMEWORK_OUTPUT_SAMPLE_RATE = 24000


class TransportClosed(Exception):
    """Raised by ``receive_message`` to signal a clean protocol-level close."""


class BaseTransport(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    websocket: WebSocket
    is_running: bool = False

    input_audio_format: AudioFormat = AudioFormat.PCM16
    input_audio_sample_rate: int = 16000
    input_audio_chunk_size: int = 1024
    input_audio_filter: Optional[BaseAudioFilter] = None

    output_audio_format: AudioFormat = AudioFormat.PCM16
    output_audio_sample_rate: int = 24000
    output_audio_chunk_size: int = 1920

    _input_transcoder: Optional[AudioTranscoder] = PrivateAttr(default=None)
    _input_audio_buffer: BufferService = PrivateAttr()
    _output_transcoder: Optional[AudioTranscoder] = PrivateAttr(default=None)
    _output_audio_buffer: BufferService = PrivateAttr()
    _session_logger: Optional[SessionLogger] = PrivateAttr(default=None)

    @property
    def logger(self) -> SessionLogger:
        if self._session_logger is None:
            self._session_logger = SessionLogger()
        return self._session_logger

    @logger.setter
    def logger(self, logger: SessionLogger) -> None:
        self._session_logger = logger

    def model_post_init(self, __context: Any) -> None:
        self._input_transcoder = build_transcoder(
            self.input_audio_format, self.input_audio_sample_rate,
            FRAMEWORK_INPUT_FORMAT, FRAMEWORK_INPUT_SAMPLE_RATE,
        )
        self._input_audio_buffer = BufferService(self.input_audio_chunk_size)
        self._output_transcoder = build_transcoder(
            FRAMEWORK_OUTPUT_FORMAT, FRAMEWORK_OUTPUT_SAMPLE_RATE,
            self.output_audio_format, self.output_audio_sample_rate,
        )
        self._output_audio_buffer = BufferService(self.output_audio_chunk_size)
        if self.input_audio_filter:
            self.input_audio_filter.logger = self.logger

    async def start(self) -> None:
        self.is_running = True

    async def stop(self) -> None:
        try:
            await self.flush_audio()
        except Exception as e:
            self.logger.debug(
                f"flush_audio during stop failed",
                error=str(e),
            )
        self.is_running = False
        if self.input_audio_filter:
            await self.input_audio_filter.cleanup()

    async def receive(self) -> AsyncIterator[Data]:
        if not self.is_running:
            raise RuntimeError(f"[{type(self).__name__}] Not running")
        try:
            while self.is_running:
                message = await self.websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                async for data in self.receive_message(message):
                    yield data
        except WebSocketDisconnect:
            self.logger.info("WebSocket disconnected")
        except TransportClosed:
            self.logger.info("Transport closed")
        except Exception as e:
            self.logger.error(
                "Error receiving message",
                error=str(e),
            )

    @abstractmethod
    def receive_message(self, message: Message) -> AsyncIterator[Data]:
        ...

    @abstractmethod
    async def send_audio(self, raw: bytes) -> None:
        """Send audio data to the client."""

    async def send_interruption(self, data: InterruptionData) -> None:
        """Signal an interruption to the client. No-op by default; override for transports that support interruptions."""

    async def send_text(self, data: TextData) -> None:
        """Send text data to the client. No-op by default; override for transports that support text."""

    async def send_transcript(self, data: TranscriptData) -> None:
        """Send a transcript entry to the client. No-op by default; override for transports that support transcripts."""

    async def send_event(self, data: EventData) -> None:
        """Send a transport-level event to the client. No-op by default; override in subclasses that support events."""

    async def send_voice_activity(self, data: VoiceActivityData) -> None:
        """Send a voice activity to the client. No-op by default; override for transports that support voice activity."""

    async def send_turn_complete(self, data: TurnCompleteData) -> None:
        """Send a turn complete event to the client. No-op by default; override for transports that support turn completion."""

    async def write_audio(self, data: AudioData) -> None:
        self._ensure_connected()
        transcoded = self._transcode_output(data.data)
        for chunk in self._output_audio_buffer.get_chunks(transcoded):
            await self.send_audio(chunk)

    async def flush_audio(self) -> None:
        if not self.is_running or self.websocket.client_state != WebSocketState.CONNECTED:
            return
        flushed = self._output_audio_buffer.flush()
        if not flushed:
            return
        await self.send_audio(flushed)

    async def _yield_audio(self, raw: bytes) -> AsyncIterator[AudioData]:
        transcoded = self._transcode_input(raw)
        for chunk in self._input_audio_buffer.get_chunks(transcoded):
            filtered = await self._filter_input(chunk)
            if filtered is None:
                continue
            yield AudioData(
                data=filtered,
                format=FRAMEWORK_INPUT_FORMAT,
                sample_rate=FRAMEWORK_INPUT_SAMPLE_RATE,
            )

    def _transcode_input(self, data: bytes) -> bytes:
        if self._input_transcoder is None:
            return data
        return self._input_transcoder.process(data)

    def _transcode_output(self, data: bytes) -> bytes:
        if self._output_transcoder is None:
            return data
        return self._output_transcoder.process(data)

    async def _filter_input(self, data: bytes) -> Optional[bytes]:
        if self.input_audio_filter is None:
            return data
        return await self.input_audio_filter.process(data)

    def _ensure_connected(self) -> None:
        if not self.is_running or self.websocket.client_state != WebSocketState.CONNECTED:
            raise RuntimeError(f"[{type(self).__name__}] Not connected")
