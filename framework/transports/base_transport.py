"""Abstract base transport — defines the interface for client communication."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, ClassVar, Optional

from fastapi import WebSocket
from pydantic import BaseModel, ConfigDict, PrivateAttr
from starlette.websockets import WebSocketState

from ..audio_transcoder import AudioTranscoder, build_transcoder
from ..buffer_service import BufferService
from ..models import AudioFormat, AudioData, TextData, TranscriptData, EventData, VoiceActivityData, Data


class BaseTransport(BaseModel, ABC):
    """Abstract base for all transports in the Gemini Live Framework.

    Subclasses declare the client's audio format/rate/chunk-size.
    The base auto-creates transcoders and buffers so that:
      - receive_message() yields PCM16 16kHz (framework input format)
      - send_audio() accepts PCM16 24kHz (framework output format)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    FRAMEWORK_INPUT_FORMAT: ClassVar[AudioFormat] = AudioFormat.PCM16
    FRAMEWORK_INPUT_SAMPLE_RATE: ClassVar[int] = 16000
    FRAMEWORK_OUTPUT_FORMAT: ClassVar[AudioFormat] = AudioFormat.PCM16
    FRAMEWORK_OUTPUT_SAMPLE_RATE: ClassVar[int] = 24000

    websocket: Optional[WebSocket] = None
    is_running: bool = False

    input_audio_format: AudioFormat = AudioFormat.PCM16
    input_audio_sample_rate: int = 16000
    input_audio_chunk_size: int = 960

    output_audio_format: AudioFormat = AudioFormat.PCM16
    output_audio_sample_rate: int = 24000
    output_audio_chunk_size: int = 960

    _input_audio_buffer: BufferService = PrivateAttr()
    _output_audio_buffer: BufferService = PrivateAttr()
    _input_transcoder: Optional[AudioTranscoder] = PrivateAttr(default=None)
    _output_transcoder: Optional[AudioTranscoder] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        self._input_audio_buffer = BufferService(self.input_audio_chunk_size)
        self._output_audio_buffer = BufferService(self.output_audio_chunk_size)

        self._input_transcoder = build_transcoder(
            self.input_audio_format, self.input_audio_sample_rate,
            self.FRAMEWORK_INPUT_FORMAT, self.FRAMEWORK_INPUT_SAMPLE_RATE,
        )
        self._output_transcoder = build_transcoder(
            self.FRAMEWORK_OUTPUT_FORMAT, self.FRAMEWORK_OUTPUT_SAMPLE_RATE,
            self.output_audio_format, self.output_audio_sample_rate,
        )

    def transcode_input(self, data: bytes) -> bytes:
        """Client format -> framework format (PCM16 16kHz)."""
        if self._input_transcoder is None:
            return data
        return self._input_transcoder.process(data)

    def transcode_output(self, data: bytes) -> bytes:
        """Framework format (PCM16 24kHz) -> client format."""
        if self._output_transcoder is None:
            return data
        return self._output_transcoder.process(data)

    async def start(self):
        """Start the transport. Override to add custom startup logic."""
        self.is_running = True

    async def stop(self):
        """Stop the transport. Override to add custom shutdown logic."""
        self.is_running = False

    @abstractmethod
    def receive_message(self) -> AsyncIterator[Data]:
        """Yield incoming messages (audio or text) as Data chunks."""
        ...

    @abstractmethod
    async def send_audio(self, data: AudioData) -> None:
        """Send audio data to the client."""
        ...

    @abstractmethod
    async def send_interruption(self) -> None:
        """Signal an interruption to the client."""
        ...

    async def send_text(self, data: TextData) -> None:
        """Send text data to the client. No-op by default; override for transports that support text."""

    async def send_transcript(self, data: TranscriptData) -> None:
        """Send a transcript entry to the client. No-op by default; override for transports that support transcripts."""

    async def send_event(self, data: EventData) -> None:
        """Send a transport-level event to the client. Override in subclasses that support events."""

    async def send_voice_activity(self, data: VoiceActivityData) -> None:
        """Send a voice activity to the client."""

    def _ensure_connected(self) -> None:
        if (
            not self.is_running
            or self.websocket is None
            or self.websocket.client_state != WebSocketState.CONNECTED
        ):
            raise RuntimeError(f"[{type(self).__name__}] Not connected")
