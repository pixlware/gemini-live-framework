from typing import AsyncIterator

import logging

from starlette.types import Message

from .base_transport import BaseTransport
from ..models import (
    Data,
    EventData,
    InterruptionData,
    TextData,
    TranscriptData,
    TurnCompleteData,
    VoiceActivityData,
)

logger = logging.getLogger(__name__)


class FastapiTransport(BaseTransport):
    async def receive_message(self, message: Message) -> AsyncIterator[Data]:
        if "bytes" in message and message["bytes"]:
            async for data in self._yield_audio(message["bytes"]):
                yield data
        elif "text" in message and message["text"]:
            yield TextData(text=message["text"])

    async def send_audio(self, raw: bytes) -> None:
        await self.websocket.send_bytes(raw)

    async def send_text(self, data: TextData) -> None:
        self._ensure_connected()
        await self.websocket.send_json({"type": "text", "data": data.model_dump(mode="json")})

    async def send_transcript(self, data: TranscriptData) -> None:
        self._ensure_connected()
        await self.websocket.send_json({"type": "transcript", "data": data.model_dump(mode="json")})

    async def send_interruption(self, data: InterruptionData) -> None:
        self._ensure_connected()
        await self.websocket.send_json({"type": "interruption", "data": data.model_dump(mode="json")})

    async def send_turn_complete(self, data: TurnCompleteData) -> None:
        self._ensure_connected()
        await self.websocket.send_json({"type": "turn_complete", "data": data.model_dump(mode="json")})

    async def send_voice_activity(self, data: VoiceActivityData) -> None:
        self._ensure_connected()
        await self.websocket.send_json({"type": "voice_activity", "data": data.model_dump(mode="json")})

    async def send_event(self, data: EventData) -> None:
        self._ensure_connected()
        await self.websocket.send_json({"type": "event", "data": data.model_dump(mode="json")})
