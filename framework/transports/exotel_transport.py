"""Exotel Voicebot Applet transport — bidirectional audio over Exotel's WebSocket protocol."""

import base64
import json
import logging
from typing import Optional, AsyncIterator

from fastapi import WebSocket
from starlette.websockets import WebSocketState, WebSocketDisconnect

from .base_transport import BaseTransport
from .audio_input_filter import AudioInputFilter
from ..models import AudioFormat, AudioData, EventData, Data

logger = logging.getLogger(__name__)


class ExotelTransport(BaseTransport):
    """Transport for Exotel's Voicebot Applet WebSocket streaming protocol.

    Exotel sends JSON text frames with an "event" field:
      - connected: handshake acknowledged
      - start:     stream metadata (contains stream_sid)
      - media:     base64-encoded audio in media.payload
      - mark:      playback-position confirmation
      - stop:      stream ended

    Outbound events sent by this transport:
      - media: base64-encoded audio
      - clear: flush the playback buffer (used on interruption)
      - mark:  request a playback-position callback
    """

    stream_sid: Optional[str] = None

    def __init__(
        self,
        websocket: WebSocket,
        input_audio_format: AudioFormat = AudioFormat.PCM16,
        input_audio_sample_rate: int = 16000,
        input_audio_chunk_size: int = 960,
        input_audio_filter: Optional[AudioInputFilter] = None,
        output_audio_format: AudioFormat = AudioFormat.PCM16,
        output_audio_sample_rate: int = 16000,
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
            raise RuntimeError("[ExotelTransport] Not running")

        try:
            while self.is_running and self.websocket is not None and self.websocket.client_state == WebSocketState.CONNECTED:
                raw = await self.websocket.receive_text()
                message = json.loads(raw)
                event = message.get("event")

                if event == "connected":
                    logger.info("[ExotelTransport] Connected event received")

                elif event == "start":
                    start_data = message.get("start", {})
                    self.stream_sid = start_data.get("stream_sid")
                    logger.info("[ExotelTransport] Stream started: %s", self.stream_sid)
                    yield EventData(event="start", metadata=start_data)

                elif event == "media":
                    payload = message.get("media", {}).get("payload")
                    if not payload:
                        continue
                    audio_bytes = base64.b64decode(payload)
                    transcoded = self.transcode_input(audio_bytes)
                    for chunk in self._input_audio_buffer.get_chunks(transcoded):
                        filtered = chunk
                        if self.input_audio_filter:
                            filtered = await self.input_audio_filter.process(chunk)
                        if not filtered:
                            continue
                        yield AudioData(
                            data=filtered,
                            format=self.FRAMEWORK_INPUT_FORMAT,
                            sample_rate=self.FRAMEWORK_INPUT_SAMPLE_RATE,
                        )

                elif event == "mark":
                    mark_info = message.get("mark", {})
                    yield EventData(event="mark", metadata=mark_info)

                elif event == "stop":
                    logger.info("[ExotelTransport] Stream stopped: %s", self.stream_sid)
                    break

        except (WebSocketDisconnect, RuntimeError):
            logger.info("[ExotelTransport] WebSocket disconnected")
        except Exception as e:
            logger.error("[ExotelTransport] Error receiving message: %s", e, exc_info=True)

    async def send_audio(self, data: AudioData) -> None:
        self._ensure_connected()
        transcoded = self.transcode_output(data.data)
        if not transcoded:
            return
        encoded = base64.b64encode(transcoded).decode("utf-8")
        await self.websocket.send_text(
            json.dumps(
                {
                    "event": "media",
                    "stream_sid": self.stream_sid,
                    "media": {"payload": encoded},
                }
            )
        )

    async def send_interruption(self) -> None:
        """Send an Exotel 'clear' event to flush the playback buffer."""
        self._ensure_connected()
        await self.websocket.send_text(
            json.dumps({"event": "clear", "stream_sid": self.stream_sid})
        )

    async def send_event(self, data: EventData) -> None:
        """Send a generic event to Exotel (e.g. mark)."""
        self._ensure_connected()
        message = {
            "event": data.event,
            "stream_sid": self.stream_sid,
        }
        if data.metadata:
            message[data.event] = data.metadata
        await self.websocket.send_text(json.dumps(message))
