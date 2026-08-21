import base64
import json
from typing import AsyncIterator, Optional

from starlette.types import Message

from .base_transport import BaseTransport, TransportClosed
from ..models import AudioFormat, Data, EventData, InterruptionData


class TwilioTransport(BaseTransport):
    input_audio_format: AudioFormat = AudioFormat.MULAW
    input_audio_sample_rate: int = 8000
    input_audio_chunk_size: int = 512
    output_audio_format: AudioFormat = AudioFormat.MULAW
    output_audio_sample_rate: int = 8000
    output_audio_chunk_size: int = 160

    stream_sid: Optional[str] = None

    async def receive_message(self, message: Message) -> AsyncIterator[Data]:
        if "text" not in message or not message["text"]:
            return

        payload = json.loads(message["text"])
        event = payload.get("event")

        if event == "connected":
            self.logger.info("[TwilioTransport] Connected event received")

        elif event == "start":
            start_data = payload.get("start", {})
            self.stream_sid = payload.get("streamSid") or start_data.get("streamSid")
            self.logger.info(f"[TwilioTransport] Stream started: {self.stream_sid}")
            yield EventData(event="start", data=start_data)

        elif event == "media":
            media_payload = payload.get("media", {}).get("payload")
            if media_payload:
                async for data in self._yield_audio(base64.b64decode(media_payload)):
                    yield data

        elif event == "mark":
            yield EventData(event="mark", data=payload.get("mark", {}))

        elif event == "stop":
            self.logger.info(f"[TwilioTransport] Stream stopped: {self.stream_sid}")
            raise TransportClosed

    async def send_audio(self, raw: bytes) -> None:
        encoded = base64.b64encode(raw).decode("utf-8")
        await self.websocket.send_text(json.dumps({
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {"payload": encoded},
        }))

    async def send_interruption(self, data: InterruptionData) -> None:
        self._ensure_connected()
        await self.websocket.send_text(json.dumps({
            "event": "clear",
            "streamSid": self.stream_sid,
        }))

    async def send_event(self, data: EventData) -> None:
        self._ensure_connected()
        message = {"event": data.event, "streamSid": self.stream_sid}
        if data.data:
            message[data.event] = data.data
        await self.websocket.send_text(json.dumps(message))
