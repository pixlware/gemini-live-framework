"""Accumulates audio chunks into fixed-size windows."""

import logging
from typing import Iterator

logger = logging.getLogger(__name__)


class BufferService:
    """Buffers incoming audio and yields complete chunks of ``chunk_size`` bytes."""

    chunk_size: int
    _buffer: bytearray

    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size
        self._buffer = bytearray()

    def get_chunks(self, audio_data: bytes) -> Iterator[bytes]:
        """Append *audio_data* and yield all complete chunks."""
        if not audio_data:
            return
        self._buffer.extend(audio_data)
        while len(self._buffer) >= self.chunk_size:
            yield bytes(self._buffer[:self.chunk_size])
            del self._buffer[:self.chunk_size]

    def reset(self):
        """Clear the buffer."""
        self._buffer.clear()
        logger.debug("[Buffer] Reset")
