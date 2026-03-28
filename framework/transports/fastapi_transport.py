from typing import AsyncIterator, Optional

from fastapi import WebSocket
from starlette.websockets import WebSocketState

from .base_transport import BaseTransport
from .audio_input_filter import AudioInputFilter
from ..models import AudioFormat, AudioData, TextData, TranscriptData, VoiceActivityData, Data


class FastapiTransport(BaseTransport):
    """Fastapi Transport for Gemini Live Framework."""

    def __init__(
        self,
        websocket: WebSocket,
        input_audio_format: AudioFormat = AudioFormat.PCM16,
        input_audio_sample_rate: int = 16000,
        input_audio_chunk_size: int = 960,
        input_audio_filter: Optional[AudioInputFilter] = None,
        output_audio_format: AudioFormat = AudioFormat.PCM16,
        output_audio_sample_rate: int = 24000,
        output_audio_chunk_size: int = 960,
    ):
        super().__init__(
            websocket=websocket,
            input_audio_format=input_audio_format,
            input_audio_sample_rate=input_audio_sample_rate,
            input_audio_chunk_size=input_audio_chunk_size,
            input_audio_filter=input_audio_filter,
            output_audio_format=output_audio_format,
            output_audio_sample_rate=output_audio_sample_rate,
            output_audio_chunk_size=output_audio_chunk_size,
        )

    async def receive_message(self) -> AsyncIterator[Data]:
        if not self.is_running:
            raise RuntimeError("[FastapiTransport] Not running")

        while self.is_running and self.websocket is not None and self.websocket.client_state == WebSocketState.CONNECTED:
            message = await self.websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"]:
                transcoded = self.transcode_input(message["bytes"])
                for chunk in self._input_audio_buffer.get_chunks(transcoded):
                    filtered = chunk
                    if self.input_audio_filter:
                        filtered = await self.input_audio_filter.process(chunk)
                    yield AudioData(
                        data=filtered,
                        format=self.FRAMEWORK_INPUT_FORMAT,
                        sample_rate=self.FRAMEWORK_INPUT_SAMPLE_RATE,
                    )

            elif "text" in message and message["text"]:
                yield TextData(text=message["text"])

    async def send_audio(self, data: AudioData) -> None:
        self._ensure_connected()
        transcoded = self.transcode_output(data.data)
        for chunk in self._output_audio_buffer.get_chunks(transcoded):
            await self.websocket.send_bytes(chunk)

    async def send_text(self, data: TextData) -> None:
        self._ensure_connected()
        await self.websocket.send_json({"type": "text", "data": data.model_dump()})

    async def send_transcript(self, data: TranscriptData) -> None:
        self._ensure_connected()
        await self.websocket.send_json({"type": "transcript", "data": data.model_dump()})

    async def send_interruption(self) -> None:
        self._ensure_connected()
        await self.websocket.send_json({"type": "interruption"})

    async def send_voice_activity(self, data: VoiceActivityData) -> None:
        self._ensure_connected()
        await self.websocket.send_json({"type": "voice_activity", "data": data.model_dump()})
